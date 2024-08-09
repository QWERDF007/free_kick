#include <any>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

class InstanceFactor
{
public:
    using FuncPtr = std::function<void(std::vector<std::any> &, std::vector<std::any> &)>;

    static InstanceFactor *GetInstance()
    {
        static InstanceFactor instance;
        return &instance;
    }

    void registerFuncPtr(const std::string &name, FuncPtr callable)
    {
        funs.insert({name, callable});
    }

    FuncPtr getFuncPtr(const std::string &name) const
    {
        auto found = funs.find(name);
        if (found == funs.end())
        {
            return nullptr;
        }
        return found->second;
    }

private:
    InstanceFactor()                                  = default;
    ~InstanceFactor()                                 = default;
    InstanceFactor(const InstanceFactor &)            = delete;
    InstanceFactor &operator=(const InstanceFactor &) = delete;

    std::map<std::string, FuncPtr> funs;
};

#define REGISTER_FUNCTION(FuncName, FuncPtr) InstanceFactor::GetInstance()->registerFuncPtr(#FuncName, FuncPtr)

void func1(std::vector<std::any> &inputs, std::vector<std::any> &outputs)
{
    std::cout << __FUNCTION__ << " " << std::endl;
}

void add(std::vector<std::any> &inputs, std::vector<std::any> &outputs)
{
    std::cout << __FUNCTION__ << " " << std::endl;
}

int main(int argc, char *argv[])
{
    InstanceFactor::GetInstance()->registerFuncPtr("func1", func1);
    InstanceFactor::GetInstance()->registerFuncPtr("add", add);

    REGISTER_FUNCTION(func1, func1);
    REGISTER_FUNCTION(add, add);

    std::vector<std::any> inputs;
    std::vector<std::any> outputs;
    InstanceFactor::GetInstance()->getFuncPtr("func1")(inputs, outputs);
    InstanceFactor::GetInstance()->getFuncPtr("add")(inputs, outputs);

    return 0;
}