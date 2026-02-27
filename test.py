import torch 
if __name__ == "__main__":
    m2 = torch.ones(6, 6, requires_grad=True)
    m3 = m2.std()
    m3.backward() 
    print(m2.grad)



