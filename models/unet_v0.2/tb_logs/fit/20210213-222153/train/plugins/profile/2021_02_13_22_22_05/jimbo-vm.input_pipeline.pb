	????cc@????cc@!????cc@	?y?'-???y?'-??!?y?'-??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6????cc@???m3?@1???)??b@A?=Ab?{??I????K??Yk??"???*	????Mc@2U
Iterator::Model::ParallelMapV2??e????!?B?:?:9@)??e????1?B?:?:9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?_??9??!?~\
0c7@)@?P?%??1???t;?3@:Preprocessing2F
Iterator::Model???T????!???%?`E@)?5Z?P??1v????1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??~j?t??!???K??8@)??o?DI??1??>)?*/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicef2?g@??!Cn??"@)f2?g@??1Cn??"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Ü?M??!KV??L@)???a?7??1v?-
??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorn??ʆ5u?!?'??7@)n??ʆ5u?1?'??7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??@?ȓ??!?ħh:@)ƊL??a?1k?1????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?y?'-??I r`#@Q??a3YX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???m3?@???m3?@!???m3?@      ??!       "	???)??b@???)??b@!???)??b@*      ??!       2	?=Ab?{???=Ab?{??!?=Ab?{??:	????K??????K??!????K??B      ??!       J	k??"???k??"???!k??"???R      ??!       Z	k??"???k??"???!k??"???b      ??!       JGPUY?y?'-??b q r`#@y??a3YX@