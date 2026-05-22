# =============================================================================
# ITDAA4-12 Project | Question 3 — Python + PostgreSQL Runner
# Run all SQL operations via psycopg2 (alternative to psql / pgAdmin)
# Author: Matshepo Tshabangu
# =============================================================================

# pip install psycopg2-binary

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# =============================================================================
# CONNECTION SETTINGS — update to match your PostgreSQL setup
# =============================================================================

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "user":     "postgres",        # update to your pg username
    "password": "your_password",   # update to your pg password
}
DB_NAME = "book_store"


# =============================================================================
# Helper
# =============================================================================

def run_query(cursor, query, params=None, fetch=False, label=""):
    """Execute a query and optionally print results."""
    print(f"\n{'─'*55}")
    if label:
        print(f"  {label}")
    print(f"{'─'*55}")
    cursor.execute(query, params)
    if fetch:
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        print("  " + " | ".join(f"{c:<25}" for c in col_names))
        print("  " + "-" * (28 * len(col_names)))
        for row in rows:
            print("  " + " | ".join(f"{str(v):<25}" for v in row))
        return rows
    else:
        print(f"  Executed ✓  (rowcount: {cursor.rowcount})")


# =============================================================================
# Q3.1 — Create database
# =============================================================================

print("=" * 55)
print("Q3.1 — Creating database 'book_store'")
print("=" * 55)

conn = psycopg2.connect(**DB_CONFIG, dbname="postgres")
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()

cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
if cur.fetchone():
    print(f"  Database '{DB_NAME}' already exists — skipping creation.")
else:
    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
    print(f"  Database '{DB_NAME}' created ✓")

cur.close()
conn.close()


# =============================================================================
# Connect to book_store and run all remaining operations
# =============================================================================

conn = psycopg2.connect(**DB_CONFIG, dbname=DB_NAME)
conn.autocommit = False
cur  = conn.cursor()

try:
    # ------------------------------------------------------------------
    # Q3.2 — Create Orders table + insert sample data
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("Q3.2 — Create Orders table & insert data")
    print("=" * 55)

    cur.execute("DROP TABLE IF EXISTS Orders CASCADE;")
    cur.execute("""
        CREATE TABLE Orders (
            Order_ID         SERIAL         PRIMARY KEY,
            Customer_ID      INTEGER        NOT NULL,
            Book_Title       VARCHAR(255)   NOT NULL,
            Quantity_Ordered INTEGER        NOT NULL CHECK (Quantity_Ordered > 0),
            Order_Date       DATE           NOT NULL,
            Total_Amount     NUMERIC(10,2)  NOT NULL CHECK (Total_Amount >= 0)
        );
    """)
    print("  Orders table created ✓")

    orders_data = [
        (501, 10, 'Data Science Essentials',       1, '2024-03-01',   450.00),
        (502, 11, 'Python for Beginners',           2, '2024-03-02',   600.00),
        (503, 10, 'Advanced SQL Queries',           1, '2024-03-05',   520.00),
        (504, 12, 'Machine Learning in Practice',   3, '2024-03-06', 1650.00),
    ]
    cur.executemany("""
        INSERT INTO Orders
            (Order_ID, Customer_ID, Book_Title, Quantity_Ordered, Order_Date, Total_Amount)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, orders_data)
    print(f"  Inserted {len(orders_data)} rows into Orders ✓")

    run_query(cur, "SELECT * FROM Orders ORDER BY Order_ID;",
              fetch=True, label="Orders table contents:")
    conn.commit()

    # ------------------------------------------------------------------
    # Q3.3 — Orders where Total_Amount > 1000 AND Quantity_Ordered >= 2
    # ------------------------------------------------------------------
    run_query(
        cur,
        """SELECT Order_ID, Customer_ID, Book_Title, Quantity_Ordered,
                  Order_Date, Total_Amount
           FROM Orders
           WHERE Total_Amount > 1000 AND Quantity_Ordered >= 2
           ORDER BY Total_Amount DESC;""",
        fetch=True,
        label="Q3.3 — Orders: Total_Amount > 1000 AND Quantity_Ordered >= 2"
    )

    # ------------------------------------------------------------------
    # Q3.4 — Insert Order 505
    # ------------------------------------------------------------------
    cur.execute("""
        INSERT INTO Orders
            (Order_ID, Customer_ID, Book_Title, Quantity_Ordered, Order_Date, Total_Amount)
        VALUES (505, 11, 'Deep Learning Fundamentals', 2, '2024-03-08', 1200.00);
    """)
    conn.commit()
    run_query(cur, "SELECT * FROM Orders WHERE Order_ID = 505;",
              fetch=True, label="Q3.4 — Inserted Order 505:")

    # ------------------------------------------------------------------
    # Q3.5 — Create Customers table + insert records
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("Q3.5 — Create Customers table & insert data")
    print("=" * 55)

    cur.execute("DROP TABLE IF EXISTS Customers CASCADE;")
    cur.execute("""
        CREATE TABLE Customers (
            Customer_ID    INTEGER      PRIMARY KEY,
            Customer_Name  VARCHAR(100) NOT NULL,
            Customer_Email VARCHAR(150) UNIQUE NOT NULL
        );
    """)
    customers_data = [
        (10, 'Lerato Maseko', 'lerato.m@example.com'),
        (11, 'Yusuf Daniels', 'yusuf.d@example.com'),
        (12, 'Nomsa Khumalo', 'nomsa.k@example.com'),
    ]
    cur.executemany("""
        INSERT INTO Customers (Customer_ID, Customer_Name, Customer_Email)
        VALUES (%s, %s, %s)
    """, customers_data)
    print(f"  Inserted {len(customers_data)} customers ✓")

    # Add FK constraint
    cur.execute("""
        ALTER TABLE Orders
        ADD CONSTRAINT fk_customer
        FOREIGN KEY (Customer_ID) REFERENCES Customers(Customer_ID) ON DELETE CASCADE;
    """)
    conn.commit()
    run_query(cur, "SELECT * FROM Customers ORDER BY Customer_ID;",
              fetch=True, label="Customers table:")

    # ------------------------------------------------------------------
    # Q3.6 — Orders by Customer_ID = 11 with customer details
    # ------------------------------------------------------------------
    run_query(
        cur,
        """SELECT c.Customer_ID, o.Order_ID, c.Customer_Name, c.Customer_Email,
                  o.Book_Title, o.Quantity_Ordered, o.Total_Amount
           FROM Orders o
           INNER JOIN Customers c ON o.Customer_ID = c.Customer_ID
           WHERE o.Customer_ID = 11
           ORDER BY o.Order_ID;""",
        fetch=True,
        label="Q3.6 — All orders by Customer_ID = 11:"
    )

    # ------------------------------------------------------------------
    # Q3.7 — Total books ordered + total amount per customer
    # ------------------------------------------------------------------
    run_query(
        cur,
        """SELECT c.Customer_ID, c.Customer_Name,
                  SUM(o.Quantity_Ordered) AS Total_Books_Ordered,
                  SUM(o.Total_Amount)     AS Total_Amount_Spent
           FROM Orders o
           INNER JOIN Customers c ON o.Customer_ID = c.Customer_ID
           GROUP BY c.Customer_ID, c.Customer_Name
           ORDER BY Total_Amount_Spent DESC;""",
        fetch=True,
        label="Q3.7 — Books ordered and amount spent per customer:"
    )

    # ------------------------------------------------------------------
    # Q3.8 — Large orders (Quantity_Ordered > 2), joined
    # ------------------------------------------------------------------
    run_query(
        cur,
        """SELECT c.Customer_ID, c.Customer_Name, o.Order_Date, o.Quantity_Ordered
           FROM Orders o
           INNER JOIN Customers c ON o.Customer_ID = c.Customer_ID
           WHERE o.Quantity_Ordered > 2
           ORDER BY o.Quantity_Ordered DESC, o.Order_Date;""",
        fetch=True,
        label="Q3.8 — Large orders (Quantity_Ordered > 2):"
    )

    conn.commit()
    print("\n✓ All Q3 operations completed successfully.")

except Exception as e:
    conn.rollback()
    print(f"\n✗ Error: {e}")
    raise
finally:
    cur.close()
    conn.close()
    print("Connection closed.")
