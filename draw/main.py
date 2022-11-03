import matplotlib.pyplot as plt
#准备绘制数据
x = [1, 2, 4, 8]
base = 4040975
s = [round(base / 4040975, 1), round(base / 2582590, 1), round(base / 1697849, 1), round(base / 1243167, 1)]
y = [4040975, 2582590, 1697849, 1243167]

plt.plot(x, s, "b", marker='D', markersize=5)
plt.xlabel("Quantity of Node")
plt.ylabel("Speedup")
plt.title("Matrix Multiplication Speedup")

for x1, y1 in zip(x, s):
    plt.text(x1, y1, str(y1), ha='center', va='bottom', fontsize=10)
#保存图片
plt.show()
