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

    num_of_emps = 0
    raise_amount = 1.04  # this is a calss varialbe

    def __init__(
        self, first, last, pay
    ):  # this dunder method will run once the instance is created

        self.first = first
        self.last = last
        self.pay = pay
        # self.email = first + "." + last + "@company.com" #remove this line for and add a email mehtod. in this way when people change first name their email will update automatticly

        Employee_new.num_of_emps += (
            1  # in this case, to count the total num of emp, we need use class var.
        )

    @property  # this is a decorator for change this motod to property. you can ue this method later with call emp.email. no () required. and the method will update the first name.
    def email(
        self,
    ):  # add this method and removed the self.email. this will allow the email update when the first name changes. without breaking the code for existing user.
        return "{}.{}@email.com".format(self.first, self.last)

    @email.setter  # this is a setting method, it allow you to set the email perperty by emp_1.email = aaa@gmail.com
    def email(self, email):
        self.new_email = email

    def __repr__(self):  # this is more meant to used for debug propose
        return "Employee('{}','{}','{})".format(self.first, self.last, self.pay)

    def __str__(
        self,
    ):  # if you dont have repr this wont work, this more meant for the normal user.
        # return '{} - {}'.format(self.fullname(), self.email)
        pass

    def __add__(self, other):
        return self.pay + other.pay

    def __len__(self):
        return len(self.fullname())

    @property
    def fullname(
        self,
    ):  # DONT FORGET THE SELF ARGUMENT, OTHERWISE GET "TypeError: fullname() takes 0 positional arguments but 1 was given"

        return "{} {}".format(self.first, self.last)

    @fullname.setter #this allow you to set the property in this way: emp.fullname = 'Gux'
    def fullname(self, name):
        first, last = name.split(" ")
        self.first = first
        self.last = last

    @fullname.deleter # this code will run when you del emp.fullname
    def fullname(self):
        print("Delete Name!")
        self.first = None
        self.last = None

    def apply_raise(
        self,
    ):  # to use the calss variable is a better solution, rather than a function

        # self.pay = int(self.pay * 1.04 )  # when you want to cahge the percentage, you may need to go through code to change several locaiton.

        # self.pay = int(self.pay * raise_amount ) # this will cause error
        self.pay = int(
            self.pay * Employee_new.raise_amount
        )  # to use the varialbe at calss level
        self.pay = int(
            self.pay * self.raise_amount
        )  # to use the variable as instance level,

    # static method don't pass any argument
    # regular method take self as argument , See above
    # class method pass cls and argument, start with a decorator like:

    @classmethod
    def set_raise_amt(
        cls, amount
    ):  # cls stand for class, it take the class itself as a variable.
        cls.raise_amount = amount

    @classmethod
    def from_string(cls, emp_str):
        first, last, pay = emp_str.split("-")
        return cls(first, last, pay)

    @staticmethod  # if not using any cls or self variable, just setup the funciton as staticmethod
    def is_workday(day):  # this not taking cls, or self as argument
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        return True


emp_1_new = Employee_new(
    "Corey", "Schafer", 10000
)  # when you create an instance, the __init will run
emp_2_new = Employee_new("Test", "User", 20000)

print(emp_1_new.email)
print(emp_2_new.email)

# print(emp_1_new.fullname())
# print(Employee_new.fullname(emp_1_new))  # do the same thing with the line above.

print(emp_1_new.pay)
emp_1_new.apply_raise()
print(emp_1_new.pay)

# emp_1_new.raise_amount
# Employee_new.raise_amount

print(
    Employee_new.raise_amount
)  # to use a varialbe , the system will check the instance first, if did not find , it will check the class.
print(emp_1_new.raise_amount)
print(emp_2_new.raise_amount)

# result

# 1.04
# 1.04
# 1.04

emp_1_new.raise_amount = 1.05

print(Employee_new.raise_amount)
print(emp_1_new.raise_amount)
print(emp_2_new.raise_amount)

# result

# 1.04
# 1.05
# 1.04

print(Employee_new.__dict__)  # check the name space of a class
print(emp_1_new.__dict__)
print(emp_2_new.__dict__)  # note emp_2_new doesn't have a raise_amount at the moment.

# {'__module__': '__main__', 'raise_amount': 1.04, '__init__': <function Employee_new.__init__ at 0x03A312B0>, 'fullname': <function Employee_new.fullname at 0x03A31268>, 'apply_raise': <function Employee_new.apply_raise at 0x03A31220>, '__dict__': <attribute '__dict__' of 'Employee_new' objects>, '__weakref__': <attribute '__weakref__' of 'Employee_new' objects>, '__doc__': None}

# {'first': 'Corey', 'last': 'Schafer', 'pay': 10816, 'email': 'Corey.Schafer@company.com', 'raise_amount': 1.05}

# {'first': 'Test', 'last': 'User', 'pay': 20000, 'email': 'Test.User@company.com'}

print(Employee_new.num_of_emps)

# result
# 2

print(Employee_new.raise_amount)
print(emp_1_new.raise_amount)
print(emp_2_new.raise_amount)

# 1.04
# 1.05
# 1.04

Employee_new.set_raise_amt(1.02)
# Employee_new.raise_amount = 1.02 #same result
# emp_1_new.set_raise_amt(1.02) you can also run a class function in an instance.

print(Employee_new.raise_amount)
print(emp_1_new.raise_amount)
print(emp_2_new.raise_amount)

# result

# 1.02
# 1.05
# 1.02


# traditional way to create the new employee with string
emp_str_4 = "John-Doe-70000"
emp_str_5 = "Steve-Smith-30000"
emp_str_6 = "Jane-Doe-90000"

first, last, pay = emp_str_4.split("-")

emp_4_new = Employee_new(first, last, pay)

print(emp_4_new.email)

# class level function to create new employee

emp_5_new = Employee_new.from_string(emp_str_5)

print(emp_5_new.email)

# some example of class methond from readl world.  from datetime.py of standard library

# Additional constructors
'''
    @classmethod #create a datetime instance from timestamp
    def fromtimestamp(cls, t):
        "Construct a date from a POSIX timestamp (like time.time())."
        y, m, d, hh, mm, ss, weekday, jday, dst = _time.localtime(t)
        return cls(y, m, d)

    @classmethod
    def today(cls):
        "Construct a date from time.time()."
        t = _time.time()
        return cls.fromtimestamp(t)

    @classmethod
    def fromordinal(cls, n):
        """Construct a date from a proleptic Gregorian ordinal.

        January 1 of year 1 is day 1.  Only the year, month and day are
        non-zero in the result.
        """
        y, m, d = _ord2ymd(n)
        return cls(y, m, d)
'''

import datetime

my_date = datetime.date(2016, 7, 10)

print(Employee_new.is_workday(my_date))

# iheriate sub class


class Developer(
    Employee_new
):  # in this case class Developer iherite all the properties form the Employee_new class, even it is blank
    raise_amount = 1.10  # this will give the developer 1.10 of raisde , the Employee_new still keep 1.04.

    def __init__(self, first, last, pay, prog_lang):
        super().__init__(
            first, last, pay
        )  # this line of code with let the child __init__ to copy the same attribute from parent's class
        # Employee_new.__init__(self,first,last,pay) #this is different way of doing same thing, i prefer the super() one
        self.prog_lang = prog_lang


class Manager(Employee_new):
    def __init__(self, first, last, pay, employees=None):
        super().__init__(first, last, pay)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees  # you dont want to pass a mutable variable ( list, dict) as an argument.

    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)

    def remove_emp(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)

    def print_emps(self):
        for emp in self.employees:
            pass
            # print("-->", emp.fullname())


# dev_1 = Developer('Chenhao', 'Guan', 999999999)
# dev_2 = Developer('Juanjuan', 'Yao', 999999999)

# print(dev_1.email)
# print(dev_2.email)

# result

# Chenhao.Guan@company.com
# Juanjuan.Yao@company.com

# print(help(Developer))

# result
# Help on class Developer in module __main__:

# class Developer(Employee_new)
#  |  Developer(first, last, pay)
#  |
#  |  Method resolution order:  # <<==== this is important
#  |      Developer
#  |      Employee_new
#  |      builtins.object
#  |
#  |  Methods inherited from Employee_new: #<<=== inheriated method
#  |
#  |  __init__(self, first, last, pay)
#  |      Initialize self.  See help(type(self)) for accurate signature.
#  |
#  |  apply_raise(self)
#  |
#  |  fullname(self)
#  |
#  |  ----------------------------------------------------------------------
#  |  Class methods inherited from Employee_new:
#  |
#  |  from_string(emp_str) from builtins.type
#  |
#  |  set_raise_amt(amount) from builtins.type
#  |
#  |  ----------------------------------------------------------------------
#  |  Static methods inherited from Employee_new:
#  |
#  |  is_workday(day)
#  |
#  |  ----------------------------------------------------------------------
#  |  Data descriptors inherited from Employee_new:
#  |
#  |  __dict__
#  |      dictionary for instance variables (if defined)
#  |
#  |  __weakref__
#  |      list of weak references to the object (if defined)
#  |
#  |  ----------------------------------------------------------------------
#  |  Data and other attributes inherited from Employee_new:
#  |
#  |  num_of_emps = 6
#  |
#  |  raise_amount = 1.02

# None

# print(dev_1.pay)
# dev_1.apply_raise()
# print(dev_1.pay)

# 999999999
# 1040399997

# to overide the raised amount for developer, jsut do this
# class Developer(Employee_new):
#     raise_amount = 1.10

# rerun the print
# got this result

# 999999999
# 1121999997

# change any thing in the child clss wont affect the parent class.

dev_3 = Developer("Xuerong", "Xu", 999999999, "Python")
dev_2 = Developer("Chenhao", "Guan", 999999999, "Scratch")

# how to add a new attribute for the child class ?
#  super().__init__(first, last, pay)

print(dev_3.email)
# print(dev_3.fullname())
print(dev_3.prog_lang)

print("\n")

man_1 = Manager("Juanjuan", "Yao", 99999, [dev_3])
print(man_1.email)
man_1.print_emps()
man_1.add_emp(dev_2)
man_1.print_emps()


print("\n")


print(
    Employee_new.num_of_emps
)  # all the emp created in the child class are all counted in parents class.

print(isinstance(man_1, Manager))  # True
print(isinstance(man_1, Employee_new))  # True
print(isinstance(man_1, Developer))  # False

print(issubclass(Manager, Employee_new))  # True
print(issubclass(Developer, Employee_new))  # True
print(issubclass(Manager, Developer))  # False


# real world example for the sub class is exception.py. check it.

# special magic / Dunder Method

print(1 + 2)
print("a" + "b")

# print(emp_1_new)

# result
# <__main__.Employee_new object at 0x00ECD340>

# after we put the __repr__, the result become
# Employee('Corey','Schafer','10816)

# after we setup the __str__, the result become
# Corey Schafer - Corey.Schafer@company.com

print(repr(emp_1_new))  # we can also access to the __repr__ in this way.
# print(str(emp_1_new))  #we can also access to the __str__ in this way.

print(emp_1_new.__repr__())
print(emp_1_new.__str__())

print(1 + 2)
print(int.__add__(1, 2))

print("a" + "b")
print(str.__add__("a", "b"))

print(len("test"))
print("test".__len__())

print(emp_1_new + emp_2_new)
# result
30816

print(len(emp_1_new))

# check here for the python documentation for special methond name (dunder) https://docs.python.org/3/reference/datamodel.html#special-method-names

# check datetime.py for realworld example.

# def __add__(self, other):
#     if isinstance(other, timedelta):
#         # for CPython compatibility, we cannot use
#         # our __class__ here, but need a real timedelta
#         return timedelta(self._days + other._days,
#                          self._seconds + other._seconds,
#                          self._microseconds + other._microseconds)
#     return NotImplemented


print(emp_1_new.first)

# resutl
# Corey

emp_1_new.first = "Jim"

print(emp_1_new.first)
print(emp_1_new.email)  # but the eamil is not updated

# result
# Jim
# Corey.Shafer@email.com

print(
    emp_1_new.email()
)  # to get the email updated, we can use the newly added email method.
# result
# Jim.Schafer@email.com

# but there are problems. 1. people using the code need to chagne their code from emp.email to emp.email()
# to solve the problem, we can add the property decorator.
print(emp_1_new.email)