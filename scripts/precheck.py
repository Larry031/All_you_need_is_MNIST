"""
训练前完整检查脚本
作者: Larry3301
功能: 在模型训练前进行全面的硬件、数据、模型和配置检查
"""

import torch
import torch.nn as nn
import psutil
import shutil
import numpy as np
import os
import sys
from datetime import datetime

class TrainingPreCheck:
    def __init__(self, model=None, dataloader=None, optimizer=None, criterion=None, input_shape=None):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.input_shape = input_shape
        self.check_results = {}
        
    def print_header(self, title):
        """打印标题"""
        print(f"\n{'='*60}")
        print(f"📋 {title}")
        print(f"{'='*60}")
    
    def print_success(self, message):
        """打印成功信息"""
        print(f"✅ {message}")
    
    def print_warning(self, message):
        """打印警告信息"""
        print(f"⚠️  {message}")
    
    def print_error(self, message):
        """打印错误信息"""
        print(f"❌ {message}")
    
    def check_system_environment(self):
        """检查系统环境"""
        self.print_header("系统环境检查")
        
        # Python环境
        self.print_success(f"Python版本: {sys.version}")
        self.print_success(f"PyTorch版本: {torch.__version__}")
        
        # 工作目录
        cwd = os.getcwd()
        self.print_success(f"工作目录: {cwd}")
        
        # 检查必要的包
        required_packages = ['torch', 'numpy', 'psutil']
        for package in required_packages:
            try:
                __import__(package)
                self.print_success(f"{package}: 已安装")
            except ImportError:
                self.print_error(f"{package}: 未安装")
    
    def check_hardware_resources(self):
        """检查硬件资源"""
        self.print_header("硬件资源检查")
        
        # GPU检查
        cuda_available = torch.cuda.is_available()
        self.print_success(f"CUDA可用: {cuda_available}")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            self.print_success(f"GPU数量: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                current_device = torch.cuda.current_device()
                status = "当前设备" if i == current_device else "可用设备"
                self.print_success(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB) - {status}")
            
            # CUDA版本
            cuda_version = torch.version.cuda
            if cuda_version:
                self.print_success(f"CUDA版本: {cuda_version}")
        else:
            self.print_warning("未检测到GPU，将在CPU上训练")
        
        # CPU和内存
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        self.print_success(f"CPU核心数: {cpu_count} (使用率: {cpu_percent}%)")
        self.print_success(f"内存: {memory.total / 1e9:.1f}GB (使用率: {memory.percent}%)")
        
        # 磁盘空间
        total, used, free = shutil.disk_usage(".")
        usage_percent = (used / total) * 100
        self.print_success(f"磁盘空间: {free // (2**30)}GB 可用 (总共: {total // (2**30)}GB, 使用率: {usage_percent:.1f}%)")
        
        if usage_percent > 90:
            self.print_warning("磁盘空间不足，建议清理空间")
    
    def check_data_pipeline(self):
        """检查数据管道"""
        if self.dataloader is None:
            self.print_warning("未提供数据加载器，跳过数据检查")
            return
            
        self.print_header("数据管道检查")
        
        dataset = self.dataloader.dataset
        self.print_success(f"数据集类型: {type(dataset).__name__}")
        self.print_success(f"数据集大小: {len(dataset):,} 样本")
        self.print_success(f"Batch大小: {self.dataloader.batch_size}")
        self.print_success(f"Batch数量: {len(self.dataloader)}")
        
        # 检查一个batch的数据
        try:
            sample_batch = next(iter(self.dataloader))
            
            if isinstance(sample_batch, (list, tuple)):
                inputs, targets = sample_batch
                self.print_success(f"输入形状: {inputs.shape}")
                self.print_success(f"标签形状: {targets.shape}")
                self.print_success(f"输入数据类型: {inputs.dtype}")
                
                # 数据范围检查
                if inputs.dtype in [torch.float32, torch.float64]:
                    self.print_success(f"输入数据范围: [{inputs.min().item():.3f}, {inputs.max().item():.3f}]")
                    if inputs.min() < -10 or inputs.max() > 10:
                        self.print_warning("输入数据范围较大，考虑归一化")
                
                # NaN/Inf检查
                if torch.isnan(inputs).any():
                    self.print_error("输入数据包含NaN值!")
                if torch.isinf(inputs).any():
                    self.print_error("输入数据包含Inf值!")
                    
            else:
                self.print_success(f"数据形状: {sample_batch.shape}")
                
        except Exception as e:
            self.print_error(f"数据加载失败: {e}")
    
    def check_model_architecture(self):
        """检查模型架构"""
        if self.model is None:
            self.print_warning("未提供模型，跳过模型检查")
            return
            
        self.print_header("模型架构检查")
        
        self.print_success(f"模型类型: {type(self.model).__name__}")
        
        # 参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        self.print_success(f"总参数量: {total_params:,}")
        self.print_success(f"可训练参数: {trainable_params:,}")
        self.print_success(f"不可训练参数: {non_trainable_params:,}")
        
        if trainable_params == 0:
            self.print_error("没有可训练的参数!")
        
        # 设备检查
        device = next(self.model.parameters()).device
        self.print_success(f"模型设备: {device}")
        
        # 前向传播测试
        if self.input_shape:
            try:
                self.model.eval()
                with torch.no_grad():
                    test_input = torch.randn(*self.input_shape).to(device)
                    output = self.model(test_input)
                    self.print_success(f"前向传播测试: {test_input.shape} -> {output.shape}")
                    
                    # 输出范围检查
                    if torch.isnan(output).any():
                        self.print_error("模型输出包含NaN值!")
                    if torch.isinf(output).any():
                        self.print_error("模型输出包含Inf值!")
                        
            except Exception as e:
                self.print_error(f"前向传播失败: {e}")
    
    def check_training_configuration(self):
        """检查训练配置"""
        self.print_header("训练配置检查")
        
        if self.optimizer:
            self.print_success(f"优化器: {type(self.optimizer).__name__}")
            lr = self.optimizer.param_groups[0]['lr']
            self.print_success(f"学习率: {lr}")
            
            if lr > 1.0:
                self.print_warning("学习率可能过高")
            elif lr < 1e-6:
                self.print_warning("学习率可能过低")
        
        if self.criterion:
            self.print_success(f"损失函数: {self.criterion.__class__.__name__}")
        
        if self.model:
            mode = "训练" if self.model.training else "评估"
            self.print_success(f"模型模式: {mode}")
    
    def check_memory_estimation(self):
        """预估内存使用"""
        if not self.model or not self.dataloader or not self.input_shape:
            return
            
        self.print_header("内存使用预估")
        
        if torch.cuda.is_available():
            # 模型参数内存
            param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
            
            # 梯度内存
            gradient_size = param_size
            
            # 优化器状态内存（Adam约为参数的2倍）
            optimizer_factor = 2 if isinstance(self.optimizer, torch.optim.Adam) else 1
            optimizer_size = param_size * optimizer_factor
            
            # 激活内存（粗略估计）
            batch_size = self.dataloader.batch_size
            activation_size = batch_size * np.prod(self.input_shape) * 4  # float32
            
            total_memory = param_size + buffer_size + gradient_size + optimizer_size + activation_size
            total_gb = total_memory / 1e9
            
            self.print_success(f"预估GPU内存使用: {total_gb:.2f} GB")
            
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if total_gb > available_memory * 0.8:
                self.print_warning("预估内存使用超过GPU显存的80%，可能遇到内存不足问题")
            else:
                self.print_success("内存预估安全")
    
    def run_complete_check(self):
        """运行完整检查"""
        print("🚀 开始训练前完整检查")
        print(f"📅 检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.check_system_environment()
        self.check_hardware_resources()
        self.check_data_pipeline()
        self.check_model_architecture()
        self.check_training_configuration()
        self.check_memory_estimation()
        
        self.print_header("检查总结")
        self.print_success("训练前检查完成!")
        
        # 最终建议
        if not torch.cuda.is_available():
            self.print_warning("建议: 使用GPU加速训练")
        
        if self.model and next(self.model.parameters()).device.type == 'cpu' and torch.cuda.is_available():
            self.print_warning("建议: 将模型移动到GPU")


# 使用示例
def example_usage():
    """使用示例"""
    # 假设你有以下组件
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # 创建虚拟数据加载器
    from torch.utils.data import DataLoader, TensorDataset
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 运行检查
    checker = TrainingPreCheck(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        input_shape=(1, 784)  # 输入形状
    )
    
    checker.run_complete_check()


if __name__ == "__main__":
    # 直接运行示例
    example_usage()
    
    # 或者使用你自己的组件
    # checker = TrainingPreCheck(
    #     model=your_model,
    #     dataloader=your_dataloader,
    #     optimizer=your_optimizer,
    #     criterion=your_criterion,
    #     input_shape=your_input_shape
    # )
    # checker.run_complete_check()