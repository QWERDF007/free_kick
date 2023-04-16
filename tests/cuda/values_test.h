#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <list>
#include <stdexcept>
#include <tuple>
#include <type_traits>

// 如果 T 不是 std::tuple 类型
template<class T>
struct IsTuple : std::false_type
{
};

// 如果 T 是 std::tuple 类型
template<class... TT>
struct IsTuple<std::tuple<TT...>> : std::true_type
{
};

// 如果 T 是一个引用类型，那么就将 T 的引用去掉，然后再判断去掉引用后的类型是否是 std::tuple 类型。
// std::decay_t 是一个 C++11 中的类型转换工具，它可以将一个类型转换为其对应的 decay 类型。
// decay 类型是指将一个类型中的 cv 限定符（const 和 volatile 限定符）和引用（& 和 &&）都去掉后得到的类型。
template<class T>
struct IsTuple<T &> : IsTuple<std::decay_t<T>>
{
};

// 如果 T 是 std::tuple 类型的右值引用
template<class T>
struct IsTuple<T &&> : IsTuple<std::decay_t<T>>
{
};

inline auto JoinTuple()
{
    return std::tuple<>();
}

// 递归函数，将多个tuple合并成一个tuple
template<class T, class... TAIL>
auto JoinTuple(T a0, TAIL &&...as);

// 递归函数，将多个tuple合并成一个tuple
template<class... TT, class... TAIL>
auto JoinTuple(std::tuple<TT...> a0, TAIL &&...as)
{
    auto rest = JoinTuple(std::forward<TAIL>(as)...);
    return std::tuple_cat(std::move(a0), std::move(rest));
}

template<class T, class... TAIL>
auto JoinTuple(T a0, TAIL &&...as)
{
    return std::tuple_cat(std::make_tuple(std::move(a0)), JoinTuple(std::forward<TAIL>(as)...));
}

template<class T>
struct Identity
{
    using type = T;
};

// 这个结构体的作用是为了在某些情况下提供默认的比较函数
struct Default
{
    bool operator<(const Default &that) const
    {
        return false;
    }

    bool operator==(const Default &that) const
    {
        return true;
    }
};

template<int IDX, class T, class U>
void ReplaceDefaultsImpl(U &out, const T &in)
{
    using SRC = typename std::tuple_element<IDX, T>::type;
    using DST = typename std::tuple_element<IDX, U>::type;

    if constexpr (!std::is_same_v<SRC, Default>)
    {
        std::get<IDX>(out) = static_cast<DST>(std::get<IDX>(in));
    }
}

template<class T, class U, size_t... IDX>
void ReplaceDefaultsImpl(U &out, const T &in, std::index_sequence<IDX...>)
{
    (ReplaceDefaultsImpl<IDX>(out, in), ...);
}

// 将输入中的默认值替换为 UU 中的值。如果输入中没有默认值，则输出与输入相同
template<class... UU, class... TT>
std::enable_if_t<std::is_default_constructible_v<std::tuple<UU...>>, std::tuple<UU...>> ReplaceDefaults(
    const std::tuple<TT...> &in)
{
    static_assert(sizeof...(TT) == sizeof...(UU));

    std::tuple<UU...> out;
    ReplaceDefaultsImpl(out, in, std::index_sequence_for<TT...>());
    return out;
}

template<class U, class... UU, class T, class... TT>
std::enable_if_t<!std::is_default_constructible_v<std::tuple<U, UU...>>, std::tuple<U, UU...>> ReplaceDefaults(
    const std::tuple<T, TT...> &in)
{
    static_assert(sizeof...(TT) == sizeof...(UU));

    if constexpr (std::is_same_v<T, Default>)
    {
        static_assert(std::is_default_constructible_v<U>,
                      "Param type must have an explicit default value as it's not default constructible");

        return JoinTuple(U{}, ReplaceDefaults<UU...>(Tail(in)));
    }
    else
    {
        return JoinTuple(U{std::get<0>(in)}, ReplaceDefaults<UU...>(Tail(in)));
    }
}

template<class... TT>
class ValueList
{
public:
    // 定义tuple_value_type为TT...合并后的tuple类型
    using tuple_value_type = decltype(JoinTuple(std::declval<TT>()...));

    // 如果tuple_value_type只有一个元素，则value_type为该元素类型，否则为tuple_value_type
    using value_type =
        typename std::conditional_t<std::tuple_size_v<tuple_value_type> == 1, std::tuple_element<0, tuple_value_type>,
                                    Identity<tuple_value_type>>::type;
    // 定义list_type为value_type的list类型
    using list_type = std::list<value_type>;

    // 定义iterator和const_iterator为list_type的iterator和const_iterator
    using iterator       = typename list_type::iterator;
    using const_iterator = typename list_type::const_iterator;
    ValueList()          = default;

    // 构造函数，接受一个initializer_list<value_type>类型的参数
    ValueList(std::initializer_list<value_type> initList)
        : m_list(std::move(initList))
    {
    }

    // 复制构造，从另一个 ValueList 构造
    template<class... UU, std::enable_if_t<!std::is_same_v<tuple_value_type, std::tuple<UU...>>, int> = 0>
    explicit ValueList(const ValueList<UU...> &that)
    {
        // 遍历that中的元素
        for (auto &v : that)
        {
            // 如果value_type是tuple类型
            if constexpr (IsTuple<value_type>::value)
            {
                // 将v合并成tuple，替换默认值，然后将结果添加到m_list中
                m_list.emplace_back(ReplaceDefaults<TT...>(JoinTuple(v)));
            }
            else
            {
                // 将v合并成tuple，获取第一个元素，替换默认值，然后将结果添加到m_list中
                m_list.emplace_back(std::get<0>(ReplaceDefaults<TT...>(JoinTuple(v))));
            }
        }
    }

    ValueList(const std::vector<value_type> &v)
    {
        m_list.insert(m_list.end(), v.begin(), v.end());
    }

    auto begin()
    {
        return m_list.begin();
    }

    auto end()
    {
        return m_list.end();
    }

    auto cbegin() const
    {
        return m_list.cbegin();
    }

    auto cend() const
    {
        return m_list.cend();
    }

    auto begin() const
    {
        return m_list.begin();
    }

    auto end() const
    {
        return m_list.end();
    }

    template<class A, class... AA>
    void emplace_back(A &&a0, AA &&...args)
    {
        m_list.emplace_back(JoinTuple(std::forward<A>(a0), std::forward<AA>(args)...));
    }

    auto insert(const_iterator it, value_type v)
    {
        return m_list.insert(it, std::move(v));
    }

    auto push_front(value_type v)
    {
        return m_list.emplace_front(std::move(v));
    }

    auto push_back(value_type v)
    {
        return m_list.emplace_back(std::move(v));
    }

    template<class X = void, std::enable_if_t<sizeof(X) != 0 && !std::is_same_v<tuple_value_type, value_type>, int> = 0>
    auto push_back(tuple_value_type v)
    {
        return std::apply([this](auto &...args) { m_list.emplace_back(args...); }, v);
    }

    void concat(ValueList &&other)
    {
        m_list.splice(m_list.end(), std::move(other.m_list));
    }

    void erase(iterator it)
    {
        m_list.erase(it);
    }

    void erase(iterator itbeg, iterator itend)
    {
        m_list.erase(itbeg, itend);
    }

    bool erase(value_type v)
    {
        bool removedAtLeastOne = false;

        for (auto it = m_list.begin(); it != m_list.end();)
        {
            it = std::find(it, m_list.end(), v);
            if (it != m_list.end())
            {
                m_list.erase(it++);
                removedAtLeastOne = true;
            }
        }

        return removedAtLeastOne;
    }

    bool operator==(const ValueList<TT...> &that) const
    {
        return m_list == that.m_list;
    }

    bool operator!=(const ValueList<TT...> &that) const
    {
        return m_list != that.m_list;
    }

    bool exists(const value_type &v) const
    {
        return std::find(m_list.begin(), m_list.end(), v) != m_list.end();
    }

    template<int... NN, class F, class... UU>
    friend ValueList<UU...> UniqueSort(F extractor, ValueList<UU...> a);

    size_t size() const
    {
        return m_list.size();
    }

private:
    // list of JoinTuple
    list_type m_list;
};

// clang-format off

#define FREE_KICK_INSTANTIATE_TEST_SUITE_P(GROUP, TEST, ...)                              \
    INSTANTIATE_TEST_SUITE_P(GROUP, TEST,                                                 \
                             ::testing::ValuesIn(ValueList(__VA_ARGS__)))

#define FREE_KICK_TEST_SUITE_P(TEST, ...)                                                 \
    static ValueList g_##TEST##_Params = ValueList(__VA_ARGS__);                          \
    class TEST : public ::testing::TestWithParam<decltype(g_##TEST##_Params)::value_type> \
    {                                                                                     \
    protected:                                                                            \
        template<int I>                                                                   \
        auto GetParamValue() const                                                        \
        {                                                                                 \
            return std::get<I>(GetParam());                                               \
        }                                                                                 \
    };                                                                                    \
    FREE_KICK_INSTANTIATE_TEST_SUITE_P(_, TEST, g_##TEST##_Params)

// clang-format on