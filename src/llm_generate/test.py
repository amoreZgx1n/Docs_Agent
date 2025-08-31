# import os 
# import litellm
# from litellm import completion

# os.environ["LITELLM_PROXY_API_KEY"] = "87da00501aaf448aa0f127bd59afc7b7.G9Xfrba1vEIB7BLH"

# messages = [{ "content": "Hello, how are you?","role": "user"}]

# # openai call
# response = completion(
#     model="litellm_proxy/glm-4.5", 
#     messages=messages,
#     api_base = "https://open.bigmodel.cn/api/paas/v4/", 
#     stream=None
# )

# print(response)

# import os
# import pandas as pd
# import pandasai as pai
# from pandasai_litellm import LiteLLM

# os.environ["LITELLM_PROXY_API_KEY"] = "87da00501aaf448aa0f127bd59afc7b7.G9Xfrba1vEIB7BLH"

# llm = LiteLLM(
#     model="litellm_proxy/glm-4.5",
#     api_base="https://open.bigmodel.cn/api/paas/v4"
# )

# # 全局配置一次即可，后续 DataFrame 都能用 .chat()
# pai.config.set({
#     "llm": llm,
#     "temperature": 0,
#     "seed": 26
# })

# # 你的数据
# df_raw = pd.DataFrame({
#     "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
#     "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
#     "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
# })

# # ✅ 新用法：把 pandas.DataFrame 包成语义 DF，然后直接 df.chat()
# df = pai.DataFrame(df_raw)
# print(df.chat("Which are the 5 happiest countries?"))

import pandas as pd
from data_analyze2 import analyze_data

data = pd.read_csv("./LLM_generate/order_details.csv")

result = analyze_data("不同种类商品的售卖数量、售卖总额、利润以及利润占比", data)

print(result)

