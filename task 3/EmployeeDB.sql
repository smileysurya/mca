CREATE DATABASE EmployeeDB;
USE EmployeeDB;

CREATE TABLE Employees (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50),
    salary INT,
    hire_date DATE
);

INSERT INTO Employees VALUES 
(1, 'Anjali', 'HR', 50000, '2022-01-10'),
(2, 'Ravi', 'IT', 75000, '2021-08-15'),
(3, 'Surya', 'Finance', 60000, '2023-02-20');

SELECT * FROM Employees;

SELECT name, department FROM Employees;

SELECT * FROM Employees WHERE department = 'IT';

SELECT * FROM Employees WHERE department = 'IT' AND salary > 70000;

SELECT * FROM Employees WHERE name LIKE '%a%';

SELECT * FROM Employees WHERE salary BETWEEN 50000 AND 70000;

SELECT * FROM Employees ORDER BY salary DESC;

SELECT * FROM Employees ORDER BY hire_date DESC LIMIT 2;

SELECT DISTINCT department FROM Employees;

SELECT name AS Employee_Name, department AS Dept FROM Employees;
