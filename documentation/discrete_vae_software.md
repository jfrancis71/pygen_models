```mermaid
classDiagram
    class VAE{
        +log_prob(x)
        +sample()
        +elbo()
        +kl_div(p, q)
        +reconstruct_log_prob(q(z|x), x)
    }
    class VAE_Analytic{
        +reconstruct_log_prob(q(z|x), x)
    }
    class VAE_Uniform{
        +reconstruct_log_prob(q(z|x), x)
    }
    class VAE_Reinforce{
        +reconstruct_log_prob(q(z|x), x)
    }
    class VAE_ReinforceBaseline{
        +reconstruct_log_prob(q(z|x), x)
    }
    VAE --> VAE_Analytic
    VAE --> VAE_Uniform
    VAE --> VAE_Reinforce
    VAE --> VAE_ReinforceBaseline
```
