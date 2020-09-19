import sqlite3

# conn = sqlite3.connect(":memory:")  # read database from memory
conn = sqlite3.connect("employee.db")  # read database file from disk


c = conn.cursor()

# create a table
# c.execute(
#     """CREATE TABLE employees (
#     first text,
#     last text,
#     pay integer
#     )
#      """
# )


# insert records into tables

# c.execute("INSERT INTO employees VALUES('Xu', 'Guan' , '800000')")
# c.execute("INSERT INTO employees VALUES('Chenhao', 'Guan' , '700000')")
conn.commit()

# get records from database
c.execute("SELECT * FROM employees WHERE last = 'Guan'")

# c.fetchone() #fetch only one matched record

# c.fetchmany(5) #fetch 5 matched results in a list

# c.fetchall() # return all results in a list

# print(c.fetchone())
#print(c.fetchall())

# remenber to commit your command when you execute
conn.commit()

# remenber to close the conntion when you finish operation
# conn.close()


# introducing another way to insert record ( a more practical way)
class Employee:
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay

    def __repr__(self):
        return "Employee( '{}', '{}', {})".format(self.first, self.last, self.pay)


# create few employees with our class
empl_1 = Employee("Junfeng", "Yao", 90000)
empl_2 = Employee("Mohan", "Fu", 40000)

#print(empl_1)

# 3 way to input new record
# c.execute("INSERT INTO employees VALUES({}, {} , {})".format(empl_1.first, empl_1.last, empl_1.pay)) # this is not a good way to input data into database because it is vulnerable to injection attatck

# c.execute("INSERT INTO employees VALUES(?, ? , ?)", (empl_1.first, empl_1.last, empl_1.pay))  # using tuple to pass in the value

# c.execute(
#     "INSERT INTO employees VALUES(:first, :last , :pay)",
#     {"first": empl_1.first, "last": empl_1.last, "pay": empl_1.pay}
# ) #this method using a dictionary to pass in the values.

# c.execute(
#     "INSERT INTO employees VALUES(:first, :last , :pay)",
#     {"first": empl_2.first, "last": empl_2.last, "pay": empl_2.pay}
# )

conn.commit()

# c.execute("SELECT * FROM employees WHERE last = 'Yao'")
# c.execute("SELECT * FROM employees WHERE last = ?", ('Yao',)) # you need to put the last comma , to make the second argumend as tuple
c.execute("SELECT * FROM employees WHERE last = :last", {"last": "Yao"})

conn.commit()

#print(c.fetchall())

conn.close()

# backt to the conneciton, when you use :memory:, it will creat a fresh new db everytime it runs in memeory.
# this method is very good for testing.


conn = sqlite3.connect(":memory:")  # read database from memory

c = conn.cursor()

conn.commit()

# when you working in the memory, you wont get an error like the table is already exist, or


# some simple routine or operating database

#create a table
c.execute(
    """CREATE TABLE employees (
    first text,
    last text,
    pay integer
    )
     """
)


def insert_emp(emp):
    with conn:  # in this way, you dont need to rembmer to close it.
        c.execute(
            "INSERT INTO employees VALUES(:first, :last, :pay)",
            {"first": emp.first, "last": emp.last, "pay": emp.pay},
        )


def get_emps_by_name(lastname):
    c.execute(
        "SELECT * FROM employees WHERE last = :last", {"last": lastname}
    )  # SELECT command dont need to be commited. so dont need with:.
    return c.fetchall()


def update_pay(emp, pay):
    with conn:
        c.execute(
            """UPDATE employees SET pay = :pay
                    WHERE first = :first AND last  = :last""",
            {"first": emp.first, "last": emp.last, "pay": pay},
        )


def remove_emp(emp):
    with conn:
        c.execute(
            """ DELETE from employees WHERE first = :first AND last  = :last""",
            {"first": emp.first, "last": emp.last}
        )

insert_emp(empl_1)

insert_emp(empl_2)

print(get_emps_by_name("Yao"))

update_pay(empl_1, 100000)

print(get_emps_by_name("Yao"))

remove_emp(empl_1)

print(get_emps_by_name("Yao"))

conn.close()