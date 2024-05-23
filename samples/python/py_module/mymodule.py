WTF = "This is a string from Python!"

def add(a, b):
    return a + b

def get_list(x, n):
    return [x] * n

class MyClass:
    def __init__(self, x):
        self.x = x 
    
    def __call__(self):
        self.x = add(self.x, self.x)
        return self.x