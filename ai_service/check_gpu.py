import torch
import torch.nn.functional as F
import sys

def check_gpu():
    print(f"Python: {sys.version}")
    print(f"Torch Version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("CUDA is NOT available.")
        return False
        
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name}")
    print(f"CUDA Version (Torch): {torch.version.cuda}")
    
    # Capability check (if available in this torch version)
    try:
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: {cap}")
    except:
        print("Could not get compute capability")

    print("\nAttempting Tensor Operation (Vector Add)...")
    try:
        x = torch.tensor([1.0, 2.0]).to(device)
        y = torch.tensor([3.0, 4.0]).to(device)
        z = x + y
        print(f"Success: {z.cpu().numpy()}")
    except Exception as e:
        print(f"FAILED Vector Add: {e}")
        return False

    print("\nAttempting BAD KERNEL check (Conv1d)...")
    try:
        # Mimic the failing operation: F.conv1d
        # Input: (Batch, In_Channels, Length)
        input = torch.randn(1, 1, 10).to(device)
        # Weight: (Out_Channels, In_Channels/Groups, Kernel_Size)
        weight = torch.randn(1, 1, 3).to(device)
        
        out = F.conv1d(input, weight)
        print("Success: Conv1d executed!")
        print(out.cpu().numpy())
    except Exception as e:
        print(f"FAILED Conv1d: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = check_gpu()
    if success:
        print("\n✅ GPU CONFIGURATION VERIFIED FOR BLACKWELL")
        sys.exit(0)
    else:
        print("\n❌ GPU CONFIGURATION FAILED")
        sys.exit(1)
