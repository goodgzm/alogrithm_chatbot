import sqlite3
import os
from utils import LOG, get_data_file_path
from typing import Tuple, List
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

class SQLite:
    def __init__(self):
        db_path = get_data_file_path("algorithms_code.db")
        exist_db = True
        if not os.path.exists(db_path):
            exist_db = False

        conn = sqlite3.connect(db_path, check_same_thread=False)

        cursor = conn.cursor()

        if exist_db == False:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS algorithms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                code TEXT NOT NULL,
                result TEXT
            )
            """)
            LOG.debug("建立表成功")
            conn.commit()
            self.conn = conn
            self.cursor = cursor

            loader = TextLoader(get_data_file_path("algorithm.txt"), encoding="UTF-8")
            document = loader.load()

            text_splitter = CharacterTextSplitter(
                    separator = r'\n\n-----\d+-----',
                    chunk_size=300,
                    chunk_overlap=0,
                    length_function = len,
                    is_separator_regex = True
                )
            
            docs = text_splitter.split_documents(document)
            for doc in docs:
                text = doc.page_content
                lines = text.strip().split('\n')
                current_section = None

                for line in lines:
                    line = line.strip()
                    if line == "[算法名称]":
                        current_section = "算法名称"
                        algorithm_name = ""
                    elif line == "[算法代码]":
                        current_section = "算法代码"
                        algorithm_code = ""
                    elif current_section == "算法名称":
                        algorithm_name += line + "\n"
                    elif current_section == "算法代码":
                        algorithm_code += line + "\n"

                algorithm_name = algorithm_name.strip()
                algorithm_code = algorithm_code.strip()
                LOG.debug(f'[algorithm_name]{algorithm_name}')
                LOG.debug(f'[algorithm_code]{algorithm_code}')
                statue = True
                statue |= self.Insert_Data_name_and_id(algorithm_name, algorithm_code)
        self.conn = conn
        self.cursor = cursor


    def Select_Data_By_Name(self, algorithm_name : str) -> Tuple[str, bool]:
        self.cursor.execute("SELECT * FROM algorithms WHERE name = ?", (algorithm_name,))
        existing_rows = self.cursor.fetchall()
        if existing_rows:
            return existing_rows[0][2], True
        else:
            return None, False

    def Select_Data_By_Id(self, algorithm_id : int) -> Tuple[str, int]:
        self.cursor.execute("SELECT * FROM algorithms WHERE id = ?", (algorithm_id,))
        existing_rows = self.cursor.fetchall()
        if existing_rows:
            if len(existing_rows[0]) > 3 and existing_rows[0][3] is not None:
                return existing_rows[0][3], 2
            elif len(existing_rows[0]) > 2:
                return existing_rows[0][2], 1
            return None, 0
        else:
            return None, 0       

    def Insert_Data_name_and_id(self, algorithm_name : str, algorithm_code : str):
        LOG.debug(f"[Insert_name]{algorithm_name} \n [Insert_code]{algorithm_code}")
        code, statue = self.Select_Data_By_Name(algorithm_name)
        if statue:
            return False
        else:
            sql = "INSERT INTO algorithms (name, code) VALUES (?, ?)"
            val = (algorithm_name, algorithm_code)
            self.cursor.execute(sql, val)
            self.conn.commit()
            return True

    def Insert_Date_result_By_ID(self, algorithm_id : int, result : str):
        code, statue = self.Select_Data_By_Id(algorithm_id)
        if statue == 1:
            self.cursor.execute("""
            UPDATE algorithms
            SET result = ?
            WHERE id = ?
            """, (result, algorithm_id))
            
            self.conn.commit()
            return True
        else:
            return False

    def Select_All_Data(self):  # 定义一个方法，用于查询所有算法数据，返回所有记录
        self.cursor.execute("SELECT * FROM algorithms") 
        all_rows = self.cursor.fetchall()
        return all_rows  

    def close(self):
        self.cursor.close()
        self.conn.close()
