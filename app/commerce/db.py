"""Commerce database (SQLite stand-in for Postgres/MySQL).

Also hosts:
- change_log + triggers  -> the source for CDC. Real CDC tails the database's
  write-ahead log (Debezium -> Kafka, DynamoDB Streams, Postgres logical
  replication) instead of triggers, but consumers see the same events.
- tickets                -> the human-in-the-loop (HITL) queue.
- feedback               -> thumbs up/down from shoppers, input for evals.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS customers (id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, price REAL NOT NULL, stock INTEGER NOT NULL, description TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(id), status TEXT NOT NULL,
    total REAL NOT NULL, carrier TEXT, tracking_number TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS order_items (order_id TEXT NOT NULL, product_id TEXT NOT NULL, qty INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS change_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, entity TEXT NOT NULL, entity_id TEXT NOT NULL, op TEXT NOT NULL,
    ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, status TEXT NOT NULL, customer_id TEXT,
    session_id TEXT, order_id TEXT, question TEXT NOT NULL, reason TEXT NOT NULL, resolution TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL, user_id TEXT NOT NULL, rating TEXT NOT NULL,
    comment TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER IF NOT EXISTS products_ai AFTER INSERT ON products
BEGIN INSERT INTO change_log(entity, entity_id, op) VALUES ('product', NEW.id, 'insert'); END;
CREATE TRIGGER IF NOT EXISTS products_au AFTER UPDATE ON products
BEGIN INSERT INTO change_log(entity, entity_id, op) VALUES ('product', NEW.id, 'update'); END;
CREATE TRIGGER IF NOT EXISTS products_ad AFTER DELETE ON products
BEGIN INSERT INTO change_log(entity, entity_id, op) VALUES ('product', OLD.id, 'delete'); END;
CREATE TRIGGER IF NOT EXISTS orders_au AFTER UPDATE ON orders
BEGIN INSERT INTO change_log(entity, entity_id, op) VALUES ('order', NEW.id, 'update'); END;
"""

SEED = """
INSERT OR IGNORE INTO customers VALUES
    ('c-alice', 'Alice Johnson', 'alice@example.com'),
    ('c-bob', 'Bob Smith', 'bob@example.com');
INSERT OR IGNORE INTO products VALUES
    ('P-100', 'Trailblazer Running Shoes', 129.99, 42, 'Lightweight trail running shoes with a grippy outsole.'),
    ('P-200', 'Aero Wireless Earbuds', 89.00, 0, 'Noise-cancelling wireless earbuds with 30h battery.'),
    ('P-300', 'Summit 30L Backpack', 74.50, 15, 'Water-resistant hiking backpack with laptop sleeve.'),
    ('P-400', 'Ultralight Rain Jacket', 249.00, 8, 'Packable waterproof rain jacket for hiking.');
INSERT OR IGNORE INTO orders VALUES
    ('ORD-1001', 'c-alice', 'shipped', 129.99, 'UPS', '1Z999AA10123456784', '2026-10-20'),
    ('ORD-1002', 'c-alice', 'delivered', 249.00, 'FedEx', '771234567890', '2026-10-02'),
    ('ORD-2001', 'c-bob', 'processing', 89.00, NULL, NULL, '2026-10-27');
INSERT OR IGNORE INTO order_items VALUES
    ('ORD-1001', 'P-100', 1), ('ORD-1002', 'P-400', 1), ('ORD-2001', 'P-200', 1);
"""


class CommerceDB:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            if conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0:
                conn.executescript(SEED)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- read models -------------------------------------------------------
    def get_order(self, order_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
            if row is None:
                return None
            items = conn.execute(
                "SELECT p.name, oi.qty FROM order_items oi JOIN products p ON p.id = oi.product_id WHERE oi.order_id = ?",
                (order_id,),
            ).fetchall()
        return {**dict(row), "items": [dict(i) for i in items]}

    def list_orders(self, customer_id: str, limit: int = 5) -> list[dict]:
        with self.connect() as conn:
            ids = conn.execute(
                "SELECT id FROM orders WHERE customer_id = ? ORDER BY created_at DESC LIMIT ?", (customer_id, limit)
            ).fetchall()
        return [self.get_order(r["id"]) for r in ids]

    def all_products(self) -> list[dict]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM products").fetchall()]

    def update_product(self, product_id: str, price: float | None = None, stock: int | None = None) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "UPDATE products SET price = COALESCE(?, price), stock = COALESCE(?, stock) WHERE id = ?",
                (price, stock, product_id),
            )
            return cur.rowcount > 0

    # --- CDC ---------------------------------------------------------------
    def max_change_seq(self) -> int:
        with self.connect() as conn:
            return conn.execute("SELECT COALESCE(MAX(seq), 0) FROM change_log").fetchone()[0]

    def changes_since(self, seq: int) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM change_log WHERE seq > ? ORDER BY seq", (seq,)).fetchall()
        return [dict(r) for r in rows]

    # --- HITL --------------------------------------------------------------
    def create_ticket(self, kind: str, status: str, customer_id: str | None, session_id: str | None,
                      question: str, reason: str, order_id: str | None = None) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO tickets(kind, status, customer_id, session_id, order_id, question, reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (kind, status, customer_id, session_id, order_id, question, reason),
            )
            return cur.lastrowid

    def list_tickets(self, status: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM tickets", ()
        if status:
            sql, args = sql + " WHERE status = ?", (status,)
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(sql + " ORDER BY id DESC", args).fetchall()]

    def resolve_ticket(self, ticket_id: int, resolution: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "UPDATE tickets SET status = 'resolved', resolution = ? WHERE id = ? AND status = 'open'",
                (resolution, ticket_id),
            )
            return cur.rowcount > 0

    def add_feedback(self, request_id: str, user_id: str, rating: str, comment: str | None) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO feedback(request_id, user_id, rating, comment) VALUES (?, ?, ?, ?)",
                (request_id, user_id, rating, comment),
            )

    def ping(self) -> bool:
        with self.connect() as conn:
            return conn.execute("SELECT 1").fetchone()[0] == 1
