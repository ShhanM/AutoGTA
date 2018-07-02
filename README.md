# Auto-drive in GTAsa

### 2018/03/18

通过窗口句柄获取工具获得GTA程序的类名和窗口名称，通过pywin32包中的win32gui.GetWindowRect() 函数获取指定窗口的坐标，为下一步指定区域截图作准备。
pywin32按键在游戏中不响应需要底层驱动级（？）控制，Winio有Python封装pywinio。

>向窗口发送键盘数据win32api.keybd_event(0x0D, hwnd, 0, 0)

### 2018/04/21

win32api.keybd_event() 方法不能直接向游戏发送键盘数据，因为大多数游戏都使用Directinput响应键盘和鼠标的输入，从github上pygta5项目中发现解决方案：
使用ctypes.windll.user32.SendInput() 方法。

### 2018/05/12

完成数据采集部分：打开游戏后可对游戏窗口区域内的指定范围内进行截图，截图使用PIL内置的ImageGrab.grab() 方法，截图时间间隔可定，截图的同时可以记录按键的情况。截图命名格式为【按键类型 + 'k' + count】。按键响应通过 win32api.GetAsyncKeyState() 接收，主要监控W、A、S、D、P键，前四个用于匹配对汽车的操控动作，最后一个用于在数据采集中出现异常情况时，可以暂停/恢复采集。

### 2018/06/18

编写以下模块

- grab_pics.py

  指定时间间隔内对游戏窗口截图，保存到 /Pics 下

- models.py

  包含 LeNet，AlexNet，VGG13，ResNet34 四个卷积网络模型，基于keras

- prepare_data.py

  加载训练数据，将图片转化为数组，对 labels 进行独热编码

- train.py

  训练模型，如训练过，直接加载 autogta.h5 文件

- control.py

  游戏控制模块，包含预测函数（使用模型预测）和控制函数（调用directkeys.py 中的 PressKey、ReleaseKey 向游戏发送数据）

