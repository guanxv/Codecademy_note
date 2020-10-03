# Python Object-Oriented Programming


class Employee:
    pass


emp_1 = Employee()
emp_2 = Employee()

print(emp_1)
print(emp_2)

# both will be created as unit instance

emp_1.first = "Corey"
emp_1.last = "Schafer"
emp_1.email = "aa@gmail.com"
emp_1.pay = 1000

emp_2.first = "Test"
emp_2.last = "User"
emp_2.email = "bb@gmail.com"
emp_2.pay = 3000

print(emp_1.email)
print(emp_2.email)

# you can set up automaticly


class Employee_new:
    def __init__(self, first, last, pay):

        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + "." + last + "@company.com"

    def fullname(self): # DONT FORGET THE SELF ARGUMENT, OTHERWISE GET "TypeError: fullname() takes 0 positional arguments but 1 was given"

        return '{} {}'.format(self.first, self.last)


emp_1_new = Employee_new(
    "Corey", "Schafer", 10000
)  # when you create an instance, the __init will run
emp_2_new = Employee_new("Test", "User", 20000)

print(emp_1_new.email)
print(emp_2_new.email)

print(emp_1_new.fullname())
print(Employee_new.fullname(emp_1_new)) # do the same thing with the line above.

