	}?;l"܂@}?;l"܂@!}?;l"܂@	g}bJ+o??g}bJ+o??!g}bJ+o??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6}?;l"܂@v?ݑ?@@1ß?͚ρ@A????@???I?|x? #??Y??I`s??*	K7?A`Ya@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ڦx\T??!??D?:C@)???????1m??4@:Preprocessing2U
Iterator::Model::ParallelMapV2*?:]???!1???s3@)*?:]???11???s3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat`>Y1\??!??K?6@)]2?????1??J?W2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??6S!??!??O?1@)??6S!??1??O?1@:Preprocessing2F
Iterator::Modelip[[x??!?@.,מ?@)	??YK??1?TC(CV(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??1v?K??!?o?4JQ@)??۞ ?}?1?o<2<?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?)?D/?x?!?2J??U@)?)?D/?x?1?2J??U@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??.?Ĭ?!??g=D@)|,}???f?1?_Д?* @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9g}bJ+o??I B??@Q?G???W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	v?ݑ?@@v?ݑ?@@!v?ݑ?@@      ??!       "	ß?͚ρ@ß?͚ρ@!ß?͚ρ@*      ??!       2	????@???????@???!????@???:	?|x? #???|x? #??!?|x? #??B      ??!       J	??I`s????I`s??!??I`s??R      ??!       Z	??I`s????I`s??!??I`s??b      ??!       JGPUYg}bJ+o??b q B??@y?G???W@