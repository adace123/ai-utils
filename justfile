upgrade-to-latest:
    #!/usr/bin/env nu
    open pyproject.toml | get tool | get poetry | get dependencies | columns | each {|dep| poetry add $"($dep)@latest"}
    poetry lock
