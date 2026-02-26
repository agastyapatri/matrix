import torch 
if __name__ == "__main__":
    m1 = torch.zeros(6, 6, requires_grad=True)
    m2 = torch.ones(6, 6, requires_grad=True)
    m3 = m1 + m2
    m3.retain_grad()
    m4 = m3.sum()
    m4.backward()
    print(m3.grad)
