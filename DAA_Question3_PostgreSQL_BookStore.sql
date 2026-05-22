-- =============================================================================
-- ITDAA4-12 Project | Question 3 (30 Marks)
-- Topic: Book Store Orders Database — PostgreSQL
-- Author: Matshepo Tshabangu
-- =============================================================================
-- Run this file in psql or pgAdmin.
-- All statements use standard PostgreSQL syntax.
-- =============================================================================


-- =============================================================================
-- Q3.1 — Create database  (2 Marks)
-- =============================================================================

-- NOTE: CREATE DATABASE cannot run inside a transaction block.
-- Run this command separately in psql before connecting to the database:

CREATE DATABASE book_store;

-- Then connect: \c book_store
-- (In pgAdmin: right-click Databases → Create → Database → name: book_store)


-- =============================================================================
-- Q3.2 — Create Orders table and insert sample data  (7 Marks)
-- =============================================================================

-- Drop table if it exists (for re-running cleanly)
DROP TABLE IF EXISTS Orders CASCADE;

CREATE TABLE Orders (
    Order_ID         SERIAL          PRIMARY KEY,   -- auto-incrementing PK
    Customer_ID      INTEGER         NOT NULL,
    Book_Title       VARCHAR(255)    NOT NULL,
    Quantity_Ordered INTEGER         NOT NULL CHECK (Quantity_Ordered > 0),
    Order_Date       DATE            NOT NULL,
    Total_Amount     NUMERIC(10, 2)  NOT NULL CHECK (Total_Amount >= 0)
);

-- Insert the 4 sample rows from the scenario
INSERT INTO Orders (Order_ID, Customer_ID, Book_Title, Quantity_Ordered, Order_Date, Total_Amount)
VALUES
    (501, 10, 'Data Science Essentials',       1, '2024-03-01',   450.00),
    (502, 11, 'Python for Beginners',           2, '2024-03-02',   600.00),
    (503, 10, 'Advanced SQL Queries',           1, '2024-03-05',   520.00),
    (504, 12, 'Machine Learning in Practice',   3, '2024-03-06', 1650.00);

-- Verify the table
SELECT * FROM Orders ORDER BY Order_ID;


-- =============================================================================
-- Q3.3 — Orders where Total_Amount > 1000 AND Quantity_Ordered >= 2  (3 Marks)
-- =============================================================================

SELECT
    Order_ID,
    Customer_ID,
    Book_Title,
    Quantity_Ordered,
    Order_Date,
    Total_Amount
FROM Orders
WHERE Total_Amount > 1000
  AND Quantity_Ordered >= 2
ORDER BY Total_Amount DESC;

-- Expected output: Order 504 (Machine Learning in Practice — R1650, qty 3)


-- =============================================================================
-- Q3.4 — Insert new row (Order 505)  (3 Marks)
-- =============================================================================

INSERT INTO Orders (Order_ID, Customer_ID, Book_Title, Quantity_Ordered, Order_Date, Total_Amount)
VALUES (505, 11, 'Deep Learning Fundamentals', 2, '2024-03-08', 1200.00);

-- Verify insertion
SELECT * FROM Orders WHERE Order_ID = 505;


-- =============================================================================
-- Q3.5 — Create Customers table and insert records  (5 Marks)
-- =============================================================================

DROP TABLE IF EXISTS Customers CASCADE;

CREATE TABLE Customers (
    Customer_ID    INTEGER       PRIMARY KEY,
    Customer_Name  VARCHAR(100)  NOT NULL,
    Customer_Email VARCHAR(150)  UNIQUE NOT NULL
);

-- Insert the 3 customers from the scenario
INSERT INTO Customers (Customer_ID, Customer_Name, Customer_Email)
VALUES
    (10, 'Lerato Maseko',  'lerato.m@example.com'),
    (11, 'Yusuf Daniels',  'yusuf.d@example.com'),
    (12, 'Nomsa Khumalo',  'nomsa.k@example.com');

-- Verify
SELECT * FROM Customers ORDER BY Customer_ID;

-- Add foreign key constraint now that Customers table exists
ALTER TABLE Orders
    ADD CONSTRAINT fk_customer
    FOREIGN KEY (Customer_ID)
    REFERENCES Customers (Customer_ID)
    ON DELETE CASCADE;


-- =============================================================================
-- Q3.6 — All orders placed by Customer_ID = 11 (joined with Customers)  (4 Marks)
-- =============================================================================

SELECT
    c.Customer_ID,
    o.Order_ID,
    c.Customer_Name,
    c.Customer_Email,
    o.Book_Title,
    o.Quantity_Ordered,
    o.Total_Amount
FROM Orders AS o
INNER JOIN Customers AS c
    ON o.Customer_ID = c.Customer_ID
WHERE o.Customer_ID = 11
ORDER BY o.Order_ID;

-- Expected: Orders 502 (Python for Beginners) and 505 (Deep Learning Fundamentals)
-- for Yusuf Daniels (yusuf.d@example.com)


-- =============================================================================
-- Q3.7 — Total books ordered and total amount spent per customer  (3 Marks)
-- =============================================================================

SELECT
    c.Customer_ID,
    c.Customer_Name,
    SUM(o.Quantity_Ordered)  AS Total_Books_Ordered,
    SUM(o.Total_Amount)      AS Total_Amount_Spent
FROM Orders AS o
INNER JOIN Customers AS c
    ON o.Customer_ID = c.Customer_ID
GROUP BY c.Customer_ID, c.Customer_Name
ORDER BY Total_Amount_Spent DESC;

-- Expected output example:
-- Customer_ID | Customer_Name  | Total_Books_Ordered | Total_Amount_Spent
-- ------------|----------------|---------------------|-------------------
--  11         | Yusuf Daniels  |          4          |       1800.00
--  12         | Nomsa Khumalo  |          3          |       1650.00
--  10         | Lerato Maseko  |          2          |        970.00


-- =============================================================================
-- Q3.8 — Large orders: Quantity_Ordered > 2, joined Orders + Customers  (3 Marks)
-- =============================================================================

SELECT
    c.Customer_ID,
    c.Customer_Name,
    o.Order_Date,
    o.Quantity_Ordered
FROM Orders AS o
INNER JOIN Customers AS c
    ON o.Customer_ID = c.Customer_ID
WHERE o.Quantity_Ordered > 2
ORDER BY o.Quantity_Ordered DESC, o.Order_Date;

-- Expected: Order 504 — Nomsa Khumalo, qty 3, 2024-03-06


-- =============================================================================
-- VERIFICATION QUERIES (bonus checks)
-- =============================================================================

-- Full joined view of all orders with customer details
SELECT
    o.Order_ID,
    c.Customer_ID,
    c.Customer_Name,
    c.Customer_Email,
    o.Book_Title,
    o.Quantity_Ordered,
    o.Order_Date,
    o.Total_Amount
FROM Orders AS o
INNER JOIN Customers AS c ON o.Customer_ID = c.Customer_ID
ORDER BY o.Order_ID;

-- Count of orders per customer
SELECT
    c.Customer_Name,
    COUNT(o.Order_ID) AS Number_of_Orders
FROM Orders AS o
JOIN Customers AS c ON o.Customer_ID = c.Customer_ID
GROUP BY c.Customer_Name
ORDER BY Number_of_Orders DESC;
