#include "table.h"

#include <sqlite3.h>
#include <sqlpp11/sqlite3/sqlite3.h>
#include <sqlpp11/sqlpp11.h>

#include <iostream>

std::vector<std::string> getAllTables(sqlpp::sqlite3::connection &conn)
{
    std::vector<std::string> tables_names;

    if (!conn.is_connected())
    {
        std::cerr << "Not connected to database" << std::endl;
        return tables_names;
    }
    auto db = conn.native_handle();

    sqlite3_stmt *stmt = nullptr;
    int           rc = sqlite3_prepare_v2(db, "SELECT name FROM sqlite_master WHERE type='table';", -1, &stmt, nullptr);
    if (rc != SQLITE_OK)
    {
        std::cerr << "Could not prepare statement: " << sqlite3_errmsg(db) << std::endl;
        return tables_names;
    }
    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
        // 取出第0列字段的值
        std::string name = (char *)sqlite3_column_text(static_cast<sqlite3_stmt *>(stmt), 0);
        tables_names.emplace_back(name);
    }
    sqlite3_finalize(stmt);
    return tables_names;
}

int main(int argc, char *argv[])
{
    try
    {
        sqlpp::sqlite3::connection_config config;
        // config.path_to_database = ":memory:"; // db on memory
        config.path_to_database = "test.db"; // db on disk
        config.flags            = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
        config.debug            = false;
        sqlpp::sqlite3::connection db(config);
        if (!db.is_connected())
        {
            std::cerr << "Could not connect to database" << std::endl;
            return 1;
        }
        auto tables_names = getAllTables(db);
        bool table_exists = false;
        for (const auto table : tables_names)
        {
            if (table == "table_to_insert")
            {
                table_exists = true;
                break;
            }
        }
        if (!table_exists)
        {
            db.execute(R"(CREATE TABLE "table_to_insert" (
                "id"	INTEGER NOT NULL,
                PRIMARY KEY("id" AUTOINCREMENT)
            );)");
        }
        std::cout << "Table created" << std::endl;
        // table::TableToInsert table;
        table::TableToInsert table;
        auto                 start = std::chrono::system_clock::now();
        for (int i = 0; i < 100; ++i)
        {
            db(insert_into(table).default_values()); // 自增插入默认值或者NULL
        }
        auto end      = std::chrono::system_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "insert 100 rows use time: " << duration << "ms" << std::endl;
        for (const auto &row : db(select(table.id).from(table).unconditionally()))
        {
            std::cout << row.id << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown exception" << std::endl;
        return 1;
    }
    return 0;
}