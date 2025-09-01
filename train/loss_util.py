import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


def sobel_filter(image):
    """使用Sobel算子对每个通道计算X和Y方向的梯度"""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_x = sobel_x.to(image.device)
    sobel_y = sobel_y.to(image.device)

    # 对每个通道分别进行卷积
    grad_x = []
    grad_y = []
    for c in range(image.shape[1]):  # 对每个通道应用Sobel卷积
        grad_x_c = F.conv2d(image[:, c:c + 1, :, :], sobel_x, padding=1)
        grad_y_c = F.conv2d(image[:, c:c + 1, :, :], sobel_y, padding=1)
        grad_x.append(grad_x_c)
        grad_y.append(grad_y_c)

    # 合并所有通道的梯度
    grad_x = torch.cat(grad_x, dim=1)
    grad_y = torch.cat(grad_y, dim=1)

    return grad_x, grad_y


def create_gaussian_kernel(kernel_size=5, sigma=1.0):
    """创建高斯平滑核"""
    kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
    center = kernel_size // 2

    sigma_tensor = torch.tensor(sigma, dtype=torch.float32)

    for i in range(kernel_size):
        for j in range(kernel_size):
            x = torch.tensor(i - center, dtype=torch.float32)
            y = torch.tensor(j - center, dtype=torch.float32)
            kernel[i, j] = torch.exp(-(x * x + y * y) / (2.0 * sigma_tensor * sigma_tensor))

    # 归一化确保核的总和为1
    kernel /= kernel.sum()
    return kernel


def harris_response(grad_x, grad_y, k=0.06, kernel_size=5, sigma=1.0):
    """
        计算Harris角点响应
    #         参数:
    #             k: Harris角点响应公式中的常数，通常取0.04-0.06
    #             gaussian_kernel_size: 高斯平滑核的大小
    #             gaussian_sigma: 高斯平滑的标准差
    """

    Ixx = grad_x ** 2
    Iyy = grad_y ** 2
    Ixy = grad_x * grad_y

    # 使用torchvision.transforms.GaussianBlur进行高斯平滑
    gaussian_blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    Ixx = gaussian_blur(Ixx)
    Iyy = gaussian_blur(Iyy)
    Ixy = gaussian_blur(Ixy)

    # gaussian_kernel = create_gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0)
    # gaussian_kernel = gaussian_kernel.to(Ixx.device)
    # Ixx = F.conv2d(Ixx, gaussian_kernel, padding=kernel_size // 2)
    # Iyy = F.conv2d(Iyy, gaussian_kernel, padding=kernel_size // 2)
    # Ixy = F.conv2d(Ixy, gaussian_kernel, padding=kernel_size // 2)

    # 计算Harris角点响应R
    det = Ixx * Iyy - Ixy ** 2
    trace = Ixx + Iyy
    R = det - k * (trace ** 2)

    return R


def threshold_corners(R, threshold=0.01):
    """通过阈值处理生成二值角点图"""
    corner_map = (R > threshold).float()
    return corner_map


def harris_corner_loss(pred_image, target_image):
    """计算Harris Corner Loss"""

    # 1. 使用Sobel算子计算梯度
    grad_x_pred, grad_y_pred = sobel_filter(pred_image)
    grad_x_target, grad_y_target = sobel_filter(target_image)

    # 2. 计算Harris响应，处理RGB三通道
    R_pred = []
    R_target = []
    for c in range(pred_image.shape[1]):  # 迭代RGB通道
        R_pred_c = harris_response(grad_x_pred[:, c:c + 1, :, :], grad_y_pred[:, c:c + 1, :, :])
        R_target_c = harris_response(grad_x_target[:, c:c + 1, :, :], grad_y_target[:, c:c + 1, :, :])
        R_pred.append(R_pred_c)
        R_target.append(R_target_c)

    # 将各通道的响应合并
    R_pred = torch.cat(R_pred, dim=1)
    R_target = torch.cat(R_target, dim=1)

    # 3. 生成二值角点图
    corner_map_pred = threshold_corners(R_pred)
    corner_map_target = threshold_corners(R_target)

    # 4. 计算L1距离作为损失
    # loss = torch.abs(corner_map_pred - corner_map_target).mean()
    loss = F.l1_loss(corner_map_pred, corner_map_target, reduction='mean')

    return loss


class HarrisCornerLoss(nn.Module):
    def __init__(self, k=0.06, gaussian_kernel_size=5, gaussian_sigma=1.0, threshold=0.01):
        """
        初始化Harris角点损失函数，确保设备一致性

        参数:
            k: Harris角点响应公式中的常数，通常取0.04-0.06
            gaussian_kernel_size: 高斯平滑核的大小
            gaussian_sigma: 高斯平滑的标准差
            threshold: 角点响应的阈值，用于生成二值角点图
        """
        super(HarrisCornerLoss, self).__init__()
        self.k = k
        self.threshold = threshold

        # 定义Sobel算子（x和y方向）
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 创建高斯核
        self.gaussian_kernel = self._create_gaussian_kernel(gaussian_kernel_size, gaussian_sigma)
        self.gaussian_kernel = self.gaussian_kernel.unsqueeze(0).unsqueeze(0)

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """创建高斯平滑核"""
        kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        center = kernel_size // 2

        sigma_tensor = torch.tensor(sigma, dtype=torch.float32)

        for i in range(kernel_size):
            for j in range(kernel_size):
                x = torch.tensor(i - center, dtype=torch.float32)
                y = torch.tensor(j - center, dtype=torch.float32)
                kernel[i, j] = torch.exp(-(x * x + y * y) / (2.0 * sigma_tensor * sigma_tensor))

        # 归一化确保核的总和为1
        kernel /= kernel.sum()
        return kernel

    def compute_harris_corners(self, x):
        """
        计算图像的Harris角点

        参数:
            x: 输入图像，形状为(batch_size, channels, height, width)
                支持RGB或灰度图像

        返回:
            corner_maps: 二值角点图，形状为(batch_size, 1, height, width)
        """

        # 确保输入是单通道图像，如果是RGB则转换为灰度
        if x.size(1) == 3:
            x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]

        # 1. 使用Sobel算子计算X和Y方向梯度
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)
        Ix = F.conv2d(x, sobel_x, padding=1)
        Iy = F.conv2d(x, sobel_y, padding=1)

        # 2. 构建结构张量并应用高斯平滑
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        gaussian_kernel = self.gaussian_kernel.to(x.device)
        Ixx = F.conv2d(Ixx, gaussian_kernel, padding=self.gaussian_kernel.size(2)//2)
        Iyy = F.conv2d(Iyy, gaussian_kernel, padding=self.gaussian_kernel.size(2)//2)
        Ixy = F.conv2d(Ixy, gaussian_kernel, padding=self.gaussian_kernel.size(2)//2)

        # 3. 计算Harris角点响应R
        det_M = Ixx * Iyy - Ixy ** 2
        trace_M = Ixx + Iyy
        R = det_M - self.k * (trace_M ** 2)

        # 4. 通过阈值处理生成二值角点图
        corners = (R > self.threshold).float()

        return corners

    def forward(self, pred, target):
        """计算预测图像和真实图像之间的Harris角点损失"""

        # 计算角点图
        pred_corners = self.compute_harris_corners(pred)
        target_corners = self.compute_harris_corners(target)

        # 5. 计算L1距离作为损失
        loss = F.l1_loss(pred_corners, target_corners, reduction='mean')

        return loss


# 测试
if __name__ == "__main__":

    # 创建测试RGB图像
    pred = torch.rand(2, 3, 256, 256)  # 假设是RGB图像
    target = torch.rand(2, 3, 256, 256)  # 假设是RGB图像

    loss = harris_corner_loss(pred, target)
    print(f"Harris Corner Loss: {loss.item()}")

    # 初始化损失函数
    harris_loss = HarrisCornerLoss()

    # 计算CPU上的损失
    loss = harris_loss(pred, target)
    print(f"Harris Corner Loss: {loss.item()}")

    # 测试CUDA（如果可用）
    if torch.cuda.is_available():
        pred = pred.cuda()
        target = target.cuda()
        # 不需要手动将模型移到CUDA，内部会自动处理
        loss = harris_corner_loss(pred, target)
        print(f"Harris Corner Loss (CUDA): {loss.item()}")
        loss = harris_loss(pred, target)
        print(f"Harris Corner Loss (CUDA): {loss.item()}")

