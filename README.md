# singularity-stack
Basic container orchestration for Singularity a la `docker stack`

## Rationale

Sometimes you have an application based on Docker containers that you need to
deploy in an academic computing environment where you are required to use
Singularity. While Singularity >= 2.4 has first-class support for persistent
instances and basic orchestration in the form of SCI-F apps, neither of these
features work with containers imported directly from DockerHub.

singularity-stack implements a basic subset of the features of `docker stack`,
namely:

- Starting a stack from a configuration stored in a `docker-compose.yml` file, a la `docker stack deploy`
- Stopping a stack, a la `docker stack rm`
- Streaming the logs from a stack's services to the console

This makes it possible to maintain only one workflow based on Docker, and
deploy it on Docker with commercial providers or with Singularity with academic
providers.

## Example

The following example runs from the example subdirectory of the repository. It
requires Python 3 and pyyaml.

    example/ % ../singularity-stack.py deploy app
    example/ % singularity instance.list
    DAEMON NAME      PID      CONTAINER IMAGE
    app.archivedb    2177     /lustre/fs19/group/icecube/jvs/singularity/mysql.simg
    app.mongo        2102     /lustre/fs19/group/icecube/jvs/singularity/mongo.simg
    example/ % ../singularity-stack.py rm app
    Stopping app.archivedb instance of /lustre/fs19/group/icecube/jvs/singularity/mysql.simg (PID=2177)
    Stopping app.mongo instance of /lustre/fs19/group/icecube/jvs/singularity/mongo.simg (PID=2102)

The databases will spew a bunch of gunk into the directory tmp. You can either
clear its contents by hand or `git clean`.

## Limitations

Singularity is not Docker, and only some of the features available in
docker-compose are implemented. Specifically, only the `services` section is
parsed, and only the following keys are used:

- `image`
- `command`
- `environment`
- `depends_on`
- `volumes`
- `extra_hosts`
- `secrets`
- `deploy`
  * `restart_policy`
    - `condition`: only 'no' and 'on-failure' are recognized
    - `delay`
    - `max_attempts`

`singularity-stack` uses instances only to ensure that all processes are killed
when the stack is taken down. It would be more elegant to invoke the Docker
entry points directly from `instance.start`, but this is not currently possible.
