
"""
やりたいこと：リストをグラフにする。
    1次元配列を渡された時、そのままグラフを作成し、表示する。
    ２次元配列の時、並べて表示する。

    グラフの表示
    グラフの保存
    表示パラメータの調整

"""

class List_to_graph:
    def __init__(self,list_x,list_y) -> None:
        self.list_x = list_x
        self.list_y = list_y
        