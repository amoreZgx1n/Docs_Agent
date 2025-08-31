import pandas as pd
import numpy as np
def execute(df, code, func_name='analyze'):
        """
        执行代码
        :param code: 代码
        :param func_name: 代码中的入口函数，主要用于获取代码执行结果，与prompt中定义的让LLM完成的代码签名一致
        :return: 代码执行结果，pd.DataFrame类型
        """
        # 在namespace中执行，不指定的话，带import语句的代码，只在exec的局部作用域中，函数调用时，无法使用这些依赖
        namespace = {'pd': pd}
        # namespace['dfs'] = [self._df.copy()]
        exec(code, namespace, namespace)
        res = namespace[func_name](df)
        # res = namespace[func_name]([df.copy()])

        if isinstance(res, pd.DataFrame):
            ret_df = res
        elif isinstance(res, pd.Series):
            ret_df = convert_series_to_dataframe(res)
        elif isinstance(res, dict):
            if res['type'] == 'dataframe':
                ret_df = res['value']
            else:
                ret_df = pd.DataFrame({'结果': [res['value']]})
        else:
            ret_df = res
        return ret_df

def convert_series_to_dataframe(data_series: pd.Series):
    index_name = data_series.index.name
    list_name = data_series.name

    has_real_index = False
    index_list = data_series.index.values

    for idx, index in enumerate(index_list):
        # if not index == idx:
        if not isinstance(index, (int, np.int64)):
            has_real_index = True
            break

    if index_name:
        try:
            data_frame = data_series.reset_index(name="值")
            data_frame.columns = [index_name, "值"]
            return data_frame
        except Exception as e:
            return data_series.to_frame()
    elif has_real_index:
        return data_series.reset_index(name="值")
    else:
        return data_series.to_frame()