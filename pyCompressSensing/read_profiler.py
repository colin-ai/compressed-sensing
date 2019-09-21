import pstats

p = pstats.Stats('MCMC_sampler.profile')
p.sort_stats('cumulative').print_stats(10)
