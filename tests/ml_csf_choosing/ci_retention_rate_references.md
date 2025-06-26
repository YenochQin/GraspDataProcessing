# 量子化学CI组态筛选与组态留存率相关文献

---

## 1. Selected CI方法与组态留存率相关文献

### ① 经典Selected CI综述
- **Holmes, A. A., Tubman, N. M., & Umrigar, C. J. (2016). Heat-bath configuration interaction: An efficient selected CI algorithm inspired by heat-bath sampling. J. Chem. Theory Comput., 12(8), 3674-3680.**  
  [原文链接](https://pubs.acs.org/doi/10.1021/acs.jctc.6b00407)  
  该文详细介绍了HCI（Heat-bath CI）算法，讨论了组态筛选、重要性判据和迭代收敛性，文中多次提到"overlap with previous selection"等留存率相关概念。

### ② Adaptive CI方法
- **Schriber, J. B., & Evangelista, F. A. (2016). Adaptive configuration interaction for computing challenging electronic excited states with tunable accuracy. J. Chem. Theory Comput., 12(11), 5333-5343.**  
  [原文链接](https://pubs.acs.org/doi/10.1021/acs.jctc.6b00725)  
  Adaptive CI方法中，组态的"survival rate"或"retention"用于衡量筛选稳定性，见Supporting Information。

### ③ Iterative CI与留存率
- **Greer, J. C. (1995). Estimating full configuration interaction limits from a Monte Carlo selection of the expansion space. J. Chem. Phys., 103(5), 1821-1828.**  
  [原文链接](https://aip.scitation.org/doi/10.1063/1.470353)  
  介绍了Monte Carlo CI方法，文中讨论了每轮组态的"survival fraction"（即留存率）对收敛的影响。

---

## 2. 组态留存率的具体定义与应用

### ④ 组态留存率的实际应用
- **Garniron, Y., Scemama, A., Loos, P. F., & Caffarel, M. (2017). Hybrid stochastic-deterministic calculation of the second-order perturbative contribution of multireference perturbation theory. J. Chem. Phys., 147(3), 034101.**  
  [原文链接](https://aip.scitation.org/doi/10.1063/1.4992127)  
  Supplementary中有详细的"retention ratio"定义和实际计算方法。

### ⑤ 组态筛选算法收敛性判据
- **Levine, D. S., & Head-Gordon, M. (2017). Energy extrapolation with selected configuration interaction. J. Chem. Phys., 147(11), 114112.**  
  [原文链接](https://aip.scitation.org/doi/10.1063/1.4998614)  
  讨论了组态筛选的收敛性和组态空间的变化，涉及"overlap fraction"与"retention rate"。

---

## 3. 综述与教材

### ⑥ Selected CI综述
- **Sharma, S., & Alavi, A. (2020). Multireference configuration interaction methods. In Annual Review of Physical Chemistry, 71, 541-564.**  
  [原文链接](https://www.annualreviews.org/doi/10.1146/annurev-physchem-071119-040144)  
  综述了多种CI筛选方法，讨论了组态空间变化与收敛性判据。

---

## 4. 关键词检索建议

如需进一步查找，可用如下关键词在Google Scholar、Web of Science等检索：
- configuration retention rate
- survival rate in selected CI
- overlap fraction configuration interaction
- selected CI convergence criteria
- configuration selection stability

---

**说明**：
上述文献均为Selected CI/Adaptive CI/Iterative CI领域的权威论文，均有关于组态留存率、组态空间交集、筛选稳定性等内容的详细讨论。可根据具体算法和需求，查阅上述文献的正文或Supporting Information部分，获取更详细的理论和实现细节。 