	?0???2?@?0???2?@!?0???2?@	0k$????0k$????!0k$????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?0???2?@????@16?!}??@Aq8??9@??IG???e??YF?@12??*	a??"?mf@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?4LkӰ?!u?7?PB@)ŪA?۽??1??=?I?@:Preprocessing2U
Iterator::Model::ParallelMapV2us??=A??!?????3@)us??=A??1?????3@:Preprocessing2F
Iterator::Model???w?̰?!e??2IB@)?ôo?1??i??0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??`?>??!?w2[w?%@)??`?>??1?w2[w?%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateU?3?Y??!f?!5@)????9???1?T?*?Y$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???죃?!?ĻO?`@)???죃?1?ĻO?`@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipx?캷"??!??ͶO@)eQ?E??1n}?=?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?- ????!P?NZ?6@)}(Ff?1mc?t?>??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no90k$????I???Z8???Q9??C??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????@????@!????@      ??!       "	6?!}??@6?!}??@!6?!}??@*      ??!       2	q8??9@??q8??9@??!q8??9@??:	G???e??G???e??!G???e??B      ??!       J	F?@12??F?@12??!F?@12??R      ??!       Z	F?@12??F?@12??!F?@12??b      ??!       JGPUY0k$????b q???Z8???y9??C??X@