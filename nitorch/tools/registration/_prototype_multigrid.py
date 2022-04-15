


class PairwiseMG:

    def __init__(self, closure, lin_solver, nb_cycles=2, max_levels=16,
                 verbose=False):
        self.closure = closure
        self.lin_solver = lin_solver
        self.nb_cycles = nb_cycles
        self.max_levels = max_levels
        self.verbose = verbose



    def solve(self):

        x = self.pyrx  # solution
        i = self.pyri  # solver
        g = self.pyrg  # gradients
        d = self.pyrd  # shape

        # Initial solution at coarsest grid
        x[-1] = i[-1]()

        # The full multigrid (FMG) solver performs lots of  "two-grid"
        # multigrid (2MG) steps, at different scales. There are therefore
        # `len(pyrh) - 1` scales at which to work.

        for n_base in reversed(range(self.nb_levels-1)):
            x[n_base] = self.prolong_x(x[n_base+1], d[n_base])

            for n_cycle in range(self.nb_cycles):
                for n in range(n_base, self.nb_levels-1):
                    x[n] = i[n](x[n])
                    res = g[n](x[n])
                    g[n+1] = self.restrict(res, d[n+1])
                    del res
                    x[n+1].zero_()

                x[-1] = ih[-1](g[-1], x[-1])

                for n in reversed(range(n_base, self.nb_levels-1)):
                    x[n] += self.prolong_g(x[n+1], d[n])
                    x[n] = ih[n](g[n], x[n])

        x = x[0]
        self.clear_data()
        return x
