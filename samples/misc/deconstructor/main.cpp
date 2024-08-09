#include <iostream>

class ClassA
{
public:
    ClassA()
    {
        std::cout << "ClassA constructor called" << std::endl;
    }

    virtual ~ClassA()
    {
        std::cout << "ClassA destructor called" << std::endl;
    }
};

class ClassB : public ClassA
{
public:
    ClassB()
    {
        std::cout << "ClassB constructor called" << std::endl;
    }

    virtual ~ClassB()
    {
        std::cout << "ClassB destructor called" << std::endl;
    }
};

class ClassC : public ClassB
{
public:
    ClassC()
    {
        std::cout << "ClassC constructor called" << std::endl;
    }

    ~ClassC()
    {
        std::cout << "ClassC destructor called" << std::endl;
    }
};

int main(int argc, char *argv[])
{
    ClassA *base = new ClassC();
    delete base;
    base = nullptr;
    return 0;
}