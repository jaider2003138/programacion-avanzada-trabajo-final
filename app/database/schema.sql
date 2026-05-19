CREATE TABLE IF NOT EXISTS app_users (
    id SERIAL PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'bodega')),
    status TEXT NOT NULL DEFAULT 'activo',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inventory_audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES app_users(id),
    user_name TEXT NOT NULL,
    user_email TEXT,
    user_role TEXT NOT NULL,
    action TEXT NOT NULL,
    product_code TEXT,
    product_name TEXT,
    category TEXT,
    quantity INTEGER,
    confidence DOUBLE PRECISION,
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
