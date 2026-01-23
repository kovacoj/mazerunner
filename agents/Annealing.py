import math
import torch


class SimulatedAnnealing(torch.optim.Optimizer):
    # An implementation of simulated annealing for the
    # traveling-salesman problem, which means optimizing
    # over permutations of a fixed set of points.
    def __init__(self, params, temperature=1.0, cooling=1e-3, min_temperature=1e-6, fix_first=True, seed=0):
        super().__init__(params, {})
        self.temperature = float(temperature)
        self.cooling = float(cooling)
        self.min_temperature = float(min_temperature)
        self.fix_first = bool(fix_first)
        self._gen = torch.Generator(device="cpu")
        if seed is not None:
            self._gen.manual_seed(int(seed))

    @torch.no_grad()
    def mutate(self):
        """
        In-place proposal: swap two positions in the route tensor.

        Supports:
          - p.ndim == 1: swap two elements (perm vector)
          - p.ndim >= 2: swap two rows along dim 0 (route as ordered cities)
        """
        for group in self.param_groups:
            for p in group["params"]:
                n = p.shape[0]
                lo = 1 if (self.fix_first and n > 1) else 0
                if n - lo < 2:
                    continue

                i = int(torch.randint(lo, n, (1,), generator=self._gen))
                j = int(torch.randint(lo, n, (1,), generator=self._gen))
                while j == i:
                    j = int(torch.randint(lo, n, (1,), generator=self._gen))

                if p.ndim == 1:
                    tmp = p[i].clone()
                    p[i] = p[j]
                    p[j] = tmp
                else:
                    tmp = p[i].clone()
                    p[i].copy_(p[j])
                    p[j].copy_(tmp)

    @torch.no_grad()
    def step(self, closure):
        """
        closure(): returns scalar route length (lower is better).
        """
        if closure is None:
            raise ValueError("SimulatedAnnealing.step requires a closure() returning a scalar loss.")

        # snapshot current params
        before = [p.detach().clone() for g in self.param_groups for p in g["params"]]
        E0 = float(closure().detach().item())

        # propose
        self.mutate()
        E1 = float(closure().detach().item())

        T = max(self.temperature, 1e-12)
        accept = (E1 <= E0) or (float(torch.rand((), generator=self._gen)) < math.exp((E0 - E1) / T))

        if not accept:
            # revert
            k = 0
            for g in self.param_groups:
                for p in g["params"]:
                    p.copy_(before[k])
                    k += 1

        self.temperature = max(self.min_temperature, self.temperature * (1.0 - self.cooling))

        return E1 if accept else E0
