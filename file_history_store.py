import os
import json
import tempfile
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from typing import Sequence
import config_data as config


def get_history(session_id):
    return FileChatMessageHistory(
        session_id,
        "./chat_history",
        max_messages=config.max_history_messages,
    )

class FileChatMessageHistory(BaseChatMessageHistory):

    def __init__(self,session_id,storage_path,max_messages=12):
        self.session_id=session_id
        self.storage_path=storage_path
        self.max_messages=max_messages
        # 完整的文件路径
        self.file_path =os.path.join(self.storage_path,self.session_id)
        # 确保文件夹存在
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)
    def add_messages(self, messages: Sequence[BaseMessage])->None:
        # Sequence序列 类似list \ tuple
        all_messages=list(self.messages) # 已有的消息列表
        all_messages.extend(messages)    # 新的和已有的融合成一个list
        #
        # new_messages=[]
        # for message in all_messages:
        #     d=message_to_dirt(message)
        #     new_messages.append(d)d
        # 只保留最近的若干条，防止历史记录和模型提示词无限增长。
        new_messages=[message_to_dict(message) for message in all_messages[-self.max_messages:]]
        # 先写临时文件，再替换正式文件，避免服务中断时留下半个 JSON 文件。
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{self.session_id}.",
            suffix=".tmp",
            dir=self.storage_path,
        )
        try:
            with os.fdopen(fd,"w",encoding="utf-8") as f:
                json.dump(new_messages,f,ensure_ascii=False)
            os.replace(temp_path, self.file_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    @property     #装饰器将message方法编程成员属性用
    def messages(self)-> list[BaseMessage]:
        # 当前文件内： list[字典]
        try:
            with open(self.file_path,"r",encoding="utf-8")as f:
                message_data= json.load(f)    # 返回值就是：list 字典
                return messages_from_dict(message_data[-self.max_messages:])
        except (FileNotFoundError,json.JSONDecodeError):        # 只捕获filenotfound，但未处理JSONDecodeError等其他异常

            """当以历史纪录文件存在但内容为空，或损坏时，例如手动清空文件或写入不完整，JSON.load(F)会抛出JSONDecoderERROR，导致系统崩溃"""

            return []

    def clear(self) -> None:
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{self.session_id}.",
            suffix=".tmp",
            dir=self.storage_path,
        )
        try:
            with os.fdopen(fd,"w",encoding="utf-8") as f:
                json.dump([],f)
            os.replace(temp_path, self.file_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


