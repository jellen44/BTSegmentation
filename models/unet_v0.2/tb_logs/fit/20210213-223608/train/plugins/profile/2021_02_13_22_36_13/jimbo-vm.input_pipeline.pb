	?c????f@?c????f@!?c????f@	?a?tjH???a?tjH??!?a?tjH??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?c????f@a???pI=@1tCSvz*c@A????9??I??<?????Y????????*	U-??O^@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?6??????!?1?y??A@)Ow?x???1Fˤ? ?9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat0?1"Qh??!?O?g?
<@)??Dׅ??1v??'ߦ6@:Preprocessing2U
Iterator::Model::ParallelMapV2d\qqTn??!TWRǰ-@)d\qqTn??1TWRǰ-@:Preprocessing2F
Iterator::Model?V'g(???!?X*?E;@)Q????ێ?1gZP??(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice.??'Hl??!?0???"@).??'Hl??1?0???"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?gB?Ē??!?iu~?.R@)W?'??1?N?Ȭz@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor1Bx?q?z?!?? Q?@)1Bx?q?z?1?? Q?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?x#?ȧ?!????[(C@) C?*qm?1U??eӶ@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?a?tjH??I??3?H0@Qڮ????T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	a???pI=@a???pI=@!a???pI=@      ??!       "	tCSvz*c@tCSvz*c@!tCSvz*c@*      ??!       2	????9??????9??!????9??:	??<???????<?????!??<?????B      ??!       J	????????????????!????????R      ??!       Z	????????????????!????????b      ??!       JGPUY?a?tjH??b q??3?H0@yڮ????T@