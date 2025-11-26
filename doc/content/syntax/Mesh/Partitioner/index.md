# Partitioner System

The Partitioner System allows the developer to control the partitioning process
to split up a mesh among two or more processors. For Marlin the [/DomainPartitioner.md]
has been developed to enable mesh partitioning along the same scheme as the parallel
FFT domain partitioning.

!syntax list /Mesh/Partitioner objects=True actions=False subsystems=False

!syntax list /Mesh/Partitioner objects=False actions=False subsystems=True

!syntax list /Mesh/Partitioner objects=False actions=True subsystems=False
