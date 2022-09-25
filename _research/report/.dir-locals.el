((latex-mode . ((eval .
                      (setq
                       TeX-master
                       (concat (project-root (project-current t))
                               "_research/report/tesis.tex")))
                (ispell-local-dictionary . "spanish"))))
