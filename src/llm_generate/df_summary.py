# df_summary.py 
# 对原始数据进行统计摘要，生成以下信息：
# 列名 描述 样本值
import pandas as pd
import io

class DF_Summary:
    def __init__(self, df: pd.DataFrame, head_num: int = 5):
        self.df = df
        self.df_head = df.head(head_num)

        # 捕获 info 输出
        buffer = io.StringIO()
        df.info(buf=buffer)
        self.df_info = buffer.getvalue()

    def get_summary(self) -> dict:
        summary = {
            "head": self.df_head.to_string(index=False),
            "info": self.df_info,
            "describe": self.df.describe().to_string(),
        }
        return summary
