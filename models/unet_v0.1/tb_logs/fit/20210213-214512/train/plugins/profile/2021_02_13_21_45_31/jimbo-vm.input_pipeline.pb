	?????@?????@!?????@	??Pʯ????Pʯ??!??Pʯ??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?????@(??G @1i;???؁@A\:?<c??I?
????Y?i?????*?~j?t'h@)       =2U
Iterator::Model::ParallelMapV27 !???!?8%q$?<@)7 !???1?8%q$?<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???e???!????M9@)?{???1?{???4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateH2?w???!?s$u?e:@)?3?c?=??1????m1@:Preprocessing2F
Iterator::Model?t???l??!E!?O??E@)N?g\W??1?\p?,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??M(D???!??~+?!@)??M(D???1??~+?!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorD?K?KƁ?!??C?@)D?K?KƁ?1??C?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipB??=?
??!??f?XL@)??w?'-|?1?D???z@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?C5%Y???!6i0?;@)f??
?f?1?%?>O???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??Pʯ??I??y?m??Q????9?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	(??G @(??G @!(??G @      ??!       "	i;???؁@i;???؁@!i;???؁@*      ??!       2	\:?<c??\:?<c??!\:?<c??:	?
?????
????!?
????B      ??!       J	?i??????i?????!?i?????R      ??!       Z	?i??????i?????!?i?????b      ??!       JGPUY??Pʯ??b q??y?m??y????9?X@