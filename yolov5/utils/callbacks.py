# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Callback utils
"""


class Callbacks:
    """
    处理YOLOv5钩子（Hooks）的所有注册回调
    """

    # 定义可用的回调钩子
    _callbacks = {
        'on_pretrain_routine_start': [],  # 预训练例程开始
        'on_pretrain_routine_end': [],    # 预训练例程结束

        'on_train_start': [],              # 训练开始
        'on_train_epoch_start': [],        # 每个训练周期开始
        'on_train_batch_start': [],        # 每个训练批次开始
        'optimizer_step': [],              # 优化器步骤
        'on_before_zero_grad': [],         # 在梯度归零前
        'on_train_batch_end': [],          # 每个训练批次结束
        'on_train_epoch_end': [],          # 每个训练周期结束

        'on_val_start': [],                # 验证开始
        'on_val_batch_start': [],          # 每个验证批次开始
        'on_val_image_end': [],            # 每个验证图像结束
        'on_val_batch_end': [],            # 每个验证批次结束
        'on_val_end': [],                  # 验证结束

        'on_fit_epoch_end': [],            # 适合 = 训练 + 验证的周期结束
        'on_model_save': [],               # 模型保存时
        'on_train_end': [],                # 训练结束

        'teardown': [],                    # 清理工作
    }

    def register_action(self, hook, name='', callback=None):
        """
        注册一个新的动作到回调钩子

        参数:
            hook: 要注册动作的回调钩子名称
            name: 动作的名称以便后续引用
            callback: 触发的回调函数
        """
        # 检查钩子是否在可用的回调中
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        # 检查回调是否是可调用的
        assert callable(callback), f"callback '{callback}' is not callable"
        # 将回调添加到指定的钩子列表中
        self._callbacks[hook].append({'name': name, 'callback': callback})

    def get_registered_actions(self, hook=None):
        """
        返回所有已注册的动作，按回调钩子分类

        参数:
            hook: 要检查的钩子名称，默认为所有
        """
        if hook:
            return self._callbacks[hook]  # 返回指定钩子的回调
        else:
            return self._callbacks  # 返回所有回调钩子

    def run(self, hook, *args, **kwargs):
        """
        遍历已注册的动作并触发所有回调

        参数:
            hook: 要检查的钩子名称
            args: 从YOLOv5接收的参数
            kwargs: 从YOLOv5接收的关键字参数
        """
        # 检查钩子是否在可用的回调中
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"

        # 遍历钩子下的所有注册回调并执行
        for logger in self._callbacks[hook]:
            logger['callback'](*args, **kwargs)  # 触发回调并传递参数

