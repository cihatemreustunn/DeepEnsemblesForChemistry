же
бƒ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02unknown8љж
Ъ
!Adam/base_model_8/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/base_model_8/dense_35/bias/v
У
5Adam/base_model_8/dense_35/bias/v/Read/ReadVariableOpReadVariableOp!Adam/base_model_8/dense_35/bias/v*
_output_shapes
:*
dtype0
Ґ
#Adam/base_model_8/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adam/base_model_8/dense_35/kernel/v
Ы
7Adam/base_model_8/dense_35/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/base_model_8/dense_35/kernel/v*
_output_shapes

:*
dtype0
Ъ
!Adam/base_model_8/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/base_model_8/dense_34/bias/v
У
5Adam/base_model_8/dense_34/bias/v/Read/ReadVariableOpReadVariableOp!Adam/base_model_8/dense_34/bias/v*
_output_shapes
:*
dtype0
Ґ
#Adam/base_model_8/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#Adam/base_model_8/dense_34/kernel/v
Ы
7Adam/base_model_8/dense_34/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/base_model_8/dense_34/kernel/v*
_output_shapes

: *
dtype0
Ъ
!Adam/base_model_8/dense_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/base_model_8/dense_33/bias/v
У
5Adam/base_model_8/dense_33/bias/v/Read/ReadVariableOpReadVariableOp!Adam/base_model_8/dense_33/bias/v*
_output_shapes
: *
dtype0
Ґ
#Adam/base_model_8/dense_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#Adam/base_model_8/dense_33/kernel/v
Ы
7Adam/base_model_8/dense_33/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/base_model_8/dense_33/kernel/v*
_output_shapes

: *
dtype0
Ъ
!Adam/base_model_8/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/base_model_8/dense_32/bias/v
У
5Adam/base_model_8/dense_32/bias/v/Read/ReadVariableOpReadVariableOp!Adam/base_model_8/dense_32/bias/v*
_output_shapes
:*
dtype0
Ґ
#Adam/base_model_8/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*4
shared_name%#Adam/base_model_8/dense_32/kernel/v
Ы
7Adam/base_model_8/dense_32/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/base_model_8/dense_32/kernel/v*
_output_shapes

:	*
dtype0
Ъ
!Adam/base_model_8/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/base_model_8/dense_35/bias/m
У
5Adam/base_model_8/dense_35/bias/m/Read/ReadVariableOpReadVariableOp!Adam/base_model_8/dense_35/bias/m*
_output_shapes
:*
dtype0
Ґ
#Adam/base_model_8/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adam/base_model_8/dense_35/kernel/m
Ы
7Adam/base_model_8/dense_35/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/base_model_8/dense_35/kernel/m*
_output_shapes

:*
dtype0
Ъ
!Adam/base_model_8/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/base_model_8/dense_34/bias/m
У
5Adam/base_model_8/dense_34/bias/m/Read/ReadVariableOpReadVariableOp!Adam/base_model_8/dense_34/bias/m*
_output_shapes
:*
dtype0
Ґ
#Adam/base_model_8/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#Adam/base_model_8/dense_34/kernel/m
Ы
7Adam/base_model_8/dense_34/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/base_model_8/dense_34/kernel/m*
_output_shapes

: *
dtype0
Ъ
!Adam/base_model_8/dense_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/base_model_8/dense_33/bias/m
У
5Adam/base_model_8/dense_33/bias/m/Read/ReadVariableOpReadVariableOp!Adam/base_model_8/dense_33/bias/m*
_output_shapes
: *
dtype0
Ґ
#Adam/base_model_8/dense_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#Adam/base_model_8/dense_33/kernel/m
Ы
7Adam/base_model_8/dense_33/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/base_model_8/dense_33/kernel/m*
_output_shapes

: *
dtype0
Ъ
!Adam/base_model_8/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/base_model_8/dense_32/bias/m
У
5Adam/base_model_8/dense_32/bias/m/Read/ReadVariableOpReadVariableOp!Adam/base_model_8/dense_32/bias/m*
_output_shapes
:*
dtype0
Ґ
#Adam/base_model_8/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*4
shared_name%#Adam/base_model_8/dense_32/kernel/m
Ы
7Adam/base_model_8/dense_32/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/base_model_8/dense_32/kernel/m*
_output_shapes

:	*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
М
base_model_8/dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebase_model_8/dense_35/bias
Е
.base_model_8/dense_35/bias/Read/ReadVariableOpReadVariableOpbase_model_8/dense_35/bias*
_output_shapes
:*
dtype0
Ф
base_model_8/dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_namebase_model_8/dense_35/kernel
Н
0base_model_8/dense_35/kernel/Read/ReadVariableOpReadVariableOpbase_model_8/dense_35/kernel*
_output_shapes

:*
dtype0
М
base_model_8/dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebase_model_8/dense_34/bias
Е
.base_model_8/dense_34/bias/Read/ReadVariableOpReadVariableOpbase_model_8/dense_34/bias*
_output_shapes
:*
dtype0
Ф
base_model_8/dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_namebase_model_8/dense_34/kernel
Н
0base_model_8/dense_34/kernel/Read/ReadVariableOpReadVariableOpbase_model_8/dense_34/kernel*
_output_shapes

: *
dtype0
М
base_model_8/dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebase_model_8/dense_33/bias
Е
.base_model_8/dense_33/bias/Read/ReadVariableOpReadVariableOpbase_model_8/dense_33/bias*
_output_shapes
: *
dtype0
Ф
base_model_8/dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_namebase_model_8/dense_33/kernel
Н
0base_model_8/dense_33/kernel/Read/ReadVariableOpReadVariableOpbase_model_8/dense_33/kernel*
_output_shapes

: *
dtype0
М
base_model_8/dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebase_model_8/dense_32/bias
Е
.base_model_8/dense_32/bias/Read/ReadVariableOpReadVariableOpbase_model_8/dense_32/bias*
_output_shapes
:*
dtype0
Ф
base_model_8/dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*-
shared_namebase_model_8/dense_32/kernel
Н
0base_model_8/dense_32/kernel/Read/ReadVariableOpReadVariableOpbase_model_8/dense_32/kernel*
_output_shapes

:	*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:€€€€€€€€€	*
dtype0*
shape:€€€€€€€€€	
ђ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1base_model_8/dense_32/kernelbase_model_8/dense_32/biasbase_model_8/dense_33/kernelbase_model_8/dense_33/biasbase_model_8/dense_34/kernelbase_model_8/dense_34/biasbase_model_8/dense_35/kernelbase_model_8/dense_35/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В */
f*R(
&__inference_signature_wrapper_11150348

NoOpNoOp
Ћ5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ж5
valueь4Bщ4 Bт4
В
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1

	dense2


dense3
output_layer
	optimizer

signatures*
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
∞
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
¶
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

kernel
bias*
¶
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

kernel
bias*
¶
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

kernel
bias*
¶
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
bias*
‘
7iter

8beta_1

9beta_2
	:decay
;learning_ratem^m_m`mambmcmdmevfvgvhvivjvkvlvm*

<serving_default* 
\V
VARIABLE_VALUEbase_model_8/dense_32/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbase_model_8/dense_32/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbase_model_8/dense_33/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbase_model_8/dense_33/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbase_model_8/dense_34/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbase_model_8/dense_34/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbase_model_8/dense_35/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbase_model_8/dense_35/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
	1

2
3*

=0*
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
У
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 

0
1*

0
1*
* 
У
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 

0
1*

0
1*
* 
У
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

Qtrace_0* 

Rtrace_0* 

0
1*

0
1*
* 
У
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

Xtrace_0* 

Ytrace_0* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
8
Z	variables
[	keras_api
	\total
	]count*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

\0
]1*

Z	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_8/dense_32/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_8/dense_32/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_8/dense_33/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_8/dense_33/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_8/dense_34/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_8/dense_34/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_8/dense_35/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_8/dense_35/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_8/dense_32/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_8/dense_32/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_8/dense_33/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_8/dense_33/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_8/dense_34/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_8/dense_34/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/base_model_8/dense_35/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE!Adam/base_model_8/dense_35/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
£
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0base_model_8/dense_32/kernel/Read/ReadVariableOp.base_model_8/dense_32/bias/Read/ReadVariableOp0base_model_8/dense_33/kernel/Read/ReadVariableOp.base_model_8/dense_33/bias/Read/ReadVariableOp0base_model_8/dense_34/kernel/Read/ReadVariableOp.base_model_8/dense_34/bias/Read/ReadVariableOp0base_model_8/dense_35/kernel/Read/ReadVariableOp.base_model_8/dense_35/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp7Adam/base_model_8/dense_32/kernel/m/Read/ReadVariableOp5Adam/base_model_8/dense_32/bias/m/Read/ReadVariableOp7Adam/base_model_8/dense_33/kernel/m/Read/ReadVariableOp5Adam/base_model_8/dense_33/bias/m/Read/ReadVariableOp7Adam/base_model_8/dense_34/kernel/m/Read/ReadVariableOp5Adam/base_model_8/dense_34/bias/m/Read/ReadVariableOp7Adam/base_model_8/dense_35/kernel/m/Read/ReadVariableOp5Adam/base_model_8/dense_35/bias/m/Read/ReadVariableOp7Adam/base_model_8/dense_32/kernel/v/Read/ReadVariableOp5Adam/base_model_8/dense_32/bias/v/Read/ReadVariableOp7Adam/base_model_8/dense_33/kernel/v/Read/ReadVariableOp5Adam/base_model_8/dense_33/bias/v/Read/ReadVariableOp7Adam/base_model_8/dense_34/kernel/v/Read/ReadVariableOp5Adam/base_model_8/dense_34/bias/v/Read/ReadVariableOp7Adam/base_model_8/dense_35/kernel/v/Read/ReadVariableOp5Adam/base_model_8/dense_35/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_save_11150595
≤	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebase_model_8/dense_32/kernelbase_model_8/dense_32/biasbase_model_8/dense_33/kernelbase_model_8/dense_33/biasbase_model_8/dense_34/kernelbase_model_8/dense_34/biasbase_model_8/dense_35/kernelbase_model_8/dense_35/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount#Adam/base_model_8/dense_32/kernel/m!Adam/base_model_8/dense_32/bias/m#Adam/base_model_8/dense_33/kernel/m!Adam/base_model_8/dense_33/bias/m#Adam/base_model_8/dense_34/kernel/m!Adam/base_model_8/dense_34/bias/m#Adam/base_model_8/dense_35/kernel/m!Adam/base_model_8/dense_35/bias/m#Adam/base_model_8/dense_32/kernel/v!Adam/base_model_8/dense_32/bias/v#Adam/base_model_8/dense_33/kernel/v!Adam/base_model_8/dense_33/bias/v#Adam/base_model_8/dense_34/kernel/v!Adam/base_model_8/dense_34/bias/v#Adam/base_model_8/dense_35/kernel/v!Adam/base_model_8/dense_35/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference__traced_restore_11150698јќ
Э

ч
F__inference_dense_34_layer_call_and_return_conditional_losses_11150192

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Щ
О
J__inference_base_model_8_layer_call_and_return_conditional_losses_11150319
input_1#
dense_32_11150298:	
dense_32_11150300:#
dense_33_11150303: 
dense_33_11150305: #
dense_34_11150308: 
dense_34_11150310:#
dense_35_11150313:
dense_35_11150315:
identityИҐ dense_32/StatefulPartitionedCallҐ dense_33/StatefulPartitionedCallҐ dense_34/StatefulPartitionedCallҐ dense_35/StatefulPartitionedCallч
 dense_32/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_32_11150298dense_32_11150300*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_32_layer_call_and_return_conditional_losses_11150158Щ
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_11150303dense_33_11150305*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_33_layer_call_and_return_conditional_losses_11150175Щ
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_11150308dense_34_11150310*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_34_layer_call_and_return_conditional_losses_11150192Щ
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_11150313dense_35_11150315*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_35_layer_call_and_return_conditional_losses_11150208x
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€“
NoOpNoOp!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€	: : : : : : : : 2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€	
!
_user_specified_name	input_1
¬E
™
!__inference__traced_save_11150595
file_prefix;
7savev2_base_model_8_dense_32_kernel_read_readvariableop9
5savev2_base_model_8_dense_32_bias_read_readvariableop;
7savev2_base_model_8_dense_33_kernel_read_readvariableop9
5savev2_base_model_8_dense_33_bias_read_readvariableop;
7savev2_base_model_8_dense_34_kernel_read_readvariableop9
5savev2_base_model_8_dense_34_bias_read_readvariableop;
7savev2_base_model_8_dense_35_kernel_read_readvariableop9
5savev2_base_model_8_dense_35_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopB
>savev2_adam_base_model_8_dense_32_kernel_m_read_readvariableop@
<savev2_adam_base_model_8_dense_32_bias_m_read_readvariableopB
>savev2_adam_base_model_8_dense_33_kernel_m_read_readvariableop@
<savev2_adam_base_model_8_dense_33_bias_m_read_readvariableopB
>savev2_adam_base_model_8_dense_34_kernel_m_read_readvariableop@
<savev2_adam_base_model_8_dense_34_bias_m_read_readvariableopB
>savev2_adam_base_model_8_dense_35_kernel_m_read_readvariableop@
<savev2_adam_base_model_8_dense_35_bias_m_read_readvariableopB
>savev2_adam_base_model_8_dense_32_kernel_v_read_readvariableop@
<savev2_adam_base_model_8_dense_32_bias_v_read_readvariableopB
>savev2_adam_base_model_8_dense_33_kernel_v_read_readvariableop@
<savev2_adam_base_model_8_dense_33_bias_v_read_readvariableopB
>savev2_adam_base_model_8_dense_34_kernel_v_read_readvariableop@
<savev2_adam_base_model_8_dense_34_bias_v_read_readvariableopB
>savev2_adam_base_model_8_dense_35_kernel_v_read_readvariableop@
<savev2_adam_base_model_8_dense_35_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: џ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*Д
valueъBч B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH≠
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B П
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_base_model_8_dense_32_kernel_read_readvariableop5savev2_base_model_8_dense_32_bias_read_readvariableop7savev2_base_model_8_dense_33_kernel_read_readvariableop5savev2_base_model_8_dense_33_bias_read_readvariableop7savev2_base_model_8_dense_34_kernel_read_readvariableop5savev2_base_model_8_dense_34_bias_read_readvariableop7savev2_base_model_8_dense_35_kernel_read_readvariableop5savev2_base_model_8_dense_35_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop>savev2_adam_base_model_8_dense_32_kernel_m_read_readvariableop<savev2_adam_base_model_8_dense_32_bias_m_read_readvariableop>savev2_adam_base_model_8_dense_33_kernel_m_read_readvariableop<savev2_adam_base_model_8_dense_33_bias_m_read_readvariableop>savev2_adam_base_model_8_dense_34_kernel_m_read_readvariableop<savev2_adam_base_model_8_dense_34_bias_m_read_readvariableop>savev2_adam_base_model_8_dense_35_kernel_m_read_readvariableop<savev2_adam_base_model_8_dense_35_bias_m_read_readvariableop>savev2_adam_base_model_8_dense_32_kernel_v_read_readvariableop<savev2_adam_base_model_8_dense_32_bias_v_read_readvariableop>savev2_adam_base_model_8_dense_33_kernel_v_read_readvariableop<savev2_adam_base_model_8_dense_33_bias_v_read_readvariableop>savev2_adam_base_model_8_dense_34_kernel_v_read_readvariableop<savev2_adam_base_model_8_dense_34_bias_v_read_readvariableop>savev2_adam_base_model_8_dense_35_kernel_v_read_readvariableop<savev2_adam_base_model_8_dense_35_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*з
_input_shapes’
“: :	:: : : :::: : : : : : : :	:: : : ::::	:: : : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
: 
Э

ч
F__inference_dense_33_layer_call_and_return_conditional_losses_11150175

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
∆
Ш
+__inference_dense_33_layer_call_fn_11150429

inputs
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_33_layer_call_and_return_conditional_losses_11150175o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Э

ч
F__inference_dense_34_layer_call_and_return_conditional_losses_11150460

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ќ	
њ
/__inference_base_model_8_layer_call_fn_11150234
input_1
unknown:	
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identityИҐStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_base_model_8_layer_call_and_return_conditional_losses_11150215o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€	
!
_user_specified_name	input_1
∆
Ш
+__inference_dense_34_layer_call_fn_11150449

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_34_layer_call_and_return_conditional_losses_11150192o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Э

ч
F__inference_dense_33_layer_call_and_return_conditional_losses_11150440

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
∆
Ш
+__inference_dense_32_layer_call_fn_11150409

inputs
unknown:	
	unknown_0:
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_32_layer_call_and_return_conditional_losses_11150158o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
Ц
Н
J__inference_base_model_8_layer_call_and_return_conditional_losses_11150215

inputs#
dense_32_11150159:	
dense_32_11150161:#
dense_33_11150176: 
dense_33_11150178: #
dense_34_11150193: 
dense_34_11150195:#
dense_35_11150209:
dense_35_11150211:
identityИҐ dense_32/StatefulPartitionedCallҐ dense_33/StatefulPartitionedCallҐ dense_34/StatefulPartitionedCallҐ dense_35/StatefulPartitionedCallц
 dense_32/StatefulPartitionedCallStatefulPartitionedCallinputsdense_32_11150159dense_32_11150161*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_32_layer_call_and_return_conditional_losses_11150158Щ
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_11150176dense_33_11150178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_33_layer_call_and_return_conditional_losses_11150175Щ
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_11150193dense_34_11150195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_34_layer_call_and_return_conditional_losses_11150192Щ
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_11150209dense_35_11150211*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_35_layer_call_and_return_conditional_losses_11150208x
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€“
NoOpNoOp!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€	: : : : : : : : 2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
Э

ч
F__inference_dense_32_layer_call_and_return_conditional_losses_11150158

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
я#
Ѕ
J__inference_base_model_8_layer_call_and_return_conditional_losses_11150400

inputs9
'dense_32_matmul_readvariableop_resource:	6
(dense_32_biasadd_readvariableop_resource:9
'dense_33_matmul_readvariableop_resource: 6
(dense_33_biasadd_readvariableop_resource: 9
'dense_34_matmul_readvariableop_resource: 6
(dense_34_biasadd_readvariableop_resource:9
'dense_35_matmul_readvariableop_resource:6
(dense_35_biasadd_readvariableop_resource:
identityИҐdense_32/BiasAdd/ReadVariableOpҐdense_32/MatMul/ReadVariableOpҐdense_33/BiasAdd/ReadVariableOpҐdense_33/MatMul/ReadVariableOpҐdense_34/BiasAdd/ReadVariableOpҐdense_34/MatMul/ReadVariableOpҐdense_35/BiasAdd/ReadVariableOpҐdense_35/MatMul/ReadVariableOpЖ
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0{
dense_32/MatMulMatMulinputs&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€b
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Р
dense_33/MatMulMatMuldense_32/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0С
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ b
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ж
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Р
dense_34/MatMulMatMuldense_33/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€b
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Р
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentitydense_35/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€“
NoOpNoOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€	: : : : : : : : 2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
…	
ч
F__inference_dense_35_layer_call_and_return_conditional_losses_11150208

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Э

ч
F__inference_dense_32_layer_call_and_return_conditional_losses_11150420

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
Э	
ґ
&__inference_signature_wrapper_11150348
input_1
unknown:	
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identityИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__wrapped_model_11150140o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€	
!
_user_specified_name	input_1
 	
Њ
/__inference_base_model_8_layer_call_fn_11150369

inputs
unknown:	
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_base_model_8_layer_call_and_return_conditional_losses_11150215o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
∆
Ш
+__inference_dense_35_layer_call_fn_11150469

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dense_35_layer_call_and_return_conditional_losses_11150208o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ъ,
л
#__inference__wrapped_model_11150140
input_1F
4base_model_8_dense_32_matmul_readvariableop_resource:	C
5base_model_8_dense_32_biasadd_readvariableop_resource:F
4base_model_8_dense_33_matmul_readvariableop_resource: C
5base_model_8_dense_33_biasadd_readvariableop_resource: F
4base_model_8_dense_34_matmul_readvariableop_resource: C
5base_model_8_dense_34_biasadd_readvariableop_resource:F
4base_model_8_dense_35_matmul_readvariableop_resource:C
5base_model_8_dense_35_biasadd_readvariableop_resource:
identityИҐ,base_model_8/dense_32/BiasAdd/ReadVariableOpҐ+base_model_8/dense_32/MatMul/ReadVariableOpҐ,base_model_8/dense_33/BiasAdd/ReadVariableOpҐ+base_model_8/dense_33/MatMul/ReadVariableOpҐ,base_model_8/dense_34/BiasAdd/ReadVariableOpҐ+base_model_8/dense_34/MatMul/ReadVariableOpҐ,base_model_8/dense_35/BiasAdd/ReadVariableOpҐ+base_model_8/dense_35/MatMul/ReadVariableOp†
+base_model_8/dense_32/MatMul/ReadVariableOpReadVariableOp4base_model_8_dense_32_matmul_readvariableop_resource*
_output_shapes

:	*
dtype0Ц
base_model_8/dense_32/MatMulMatMulinput_13base_model_8/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
,base_model_8/dense_32/BiasAdd/ReadVariableOpReadVariableOp5base_model_8_dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Є
base_model_8/dense_32/BiasAddBiasAdd&base_model_8/dense_32/MatMul:product:04base_model_8/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€|
base_model_8/dense_32/ReluRelu&base_model_8/dense_32/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€†
+base_model_8/dense_33/MatMul/ReadVariableOpReadVariableOp4base_model_8_dense_33_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
base_model_8/dense_33/MatMulMatMul(base_model_8/dense_32/Relu:activations:03base_model_8/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ю
,base_model_8/dense_33/BiasAdd/ReadVariableOpReadVariableOp5base_model_8_dense_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Є
base_model_8/dense_33/BiasAddBiasAdd&base_model_8/dense_33/MatMul:product:04base_model_8/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ |
base_model_8/dense_33/ReluRelu&base_model_8/dense_33/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ †
+base_model_8/dense_34/MatMul/ReadVariableOpReadVariableOp4base_model_8_dense_34_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
base_model_8/dense_34/MatMulMatMul(base_model_8/dense_33/Relu:activations:03base_model_8/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
,base_model_8/dense_34/BiasAdd/ReadVariableOpReadVariableOp5base_model_8_dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Є
base_model_8/dense_34/BiasAddBiasAdd&base_model_8/dense_34/MatMul:product:04base_model_8/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€|
base_model_8/dense_34/ReluRelu&base_model_8/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€†
+base_model_8/dense_35/MatMul/ReadVariableOpReadVariableOp4base_model_8_dense_35_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ј
base_model_8/dense_35/MatMulMatMul(base_model_8/dense_34/Relu:activations:03base_model_8/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
,base_model_8/dense_35/BiasAdd/ReadVariableOpReadVariableOp5base_model_8_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Є
base_model_8/dense_35/BiasAddBiasAdd&base_model_8/dense_35/MatMul:product:04base_model_8/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€u
IdentityIdentity&base_model_8/dense_35/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ї
NoOpNoOp-^base_model_8/dense_32/BiasAdd/ReadVariableOp,^base_model_8/dense_32/MatMul/ReadVariableOp-^base_model_8/dense_33/BiasAdd/ReadVariableOp,^base_model_8/dense_33/MatMul/ReadVariableOp-^base_model_8/dense_34/BiasAdd/ReadVariableOp,^base_model_8/dense_34/MatMul/ReadVariableOp-^base_model_8/dense_35/BiasAdd/ReadVariableOp,^base_model_8/dense_35/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€	: : : : : : : : 2\
,base_model_8/dense_32/BiasAdd/ReadVariableOp,base_model_8/dense_32/BiasAdd/ReadVariableOp2Z
+base_model_8/dense_32/MatMul/ReadVariableOp+base_model_8/dense_32/MatMul/ReadVariableOp2\
,base_model_8/dense_33/BiasAdd/ReadVariableOp,base_model_8/dense_33/BiasAdd/ReadVariableOp2Z
+base_model_8/dense_33/MatMul/ReadVariableOp+base_model_8/dense_33/MatMul/ReadVariableOp2\
,base_model_8/dense_34/BiasAdd/ReadVariableOp,base_model_8/dense_34/BiasAdd/ReadVariableOp2Z
+base_model_8/dense_34/MatMul/ReadVariableOp+base_model_8/dense_34/MatMul/ReadVariableOp2\
,base_model_8/dense_35/BiasAdd/ReadVariableOp,base_model_8/dense_35/BiasAdd/ReadVariableOp2Z
+base_model_8/dense_35/MatMul/ReadVariableOp+base_model_8/dense_35/MatMul/ReadVariableOp:P L
'
_output_shapes
:€€€€€€€€€	
!
_user_specified_name	input_1
…	
ч
F__inference_dense_35_layer_call_and_return_conditional_losses_11150479

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Л
К
$__inference__traced_restore_11150698
file_prefix?
-assignvariableop_base_model_8_dense_32_kernel:	;
-assignvariableop_1_base_model_8_dense_32_bias:A
/assignvariableop_2_base_model_8_dense_33_kernel: ;
-assignvariableop_3_base_model_8_dense_33_bias: A
/assignvariableop_4_base_model_8_dense_34_kernel: ;
-assignvariableop_5_base_model_8_dense_34_bias:A
/assignvariableop_6_base_model_8_dense_35_kernel:;
-assignvariableop_7_base_model_8_dense_35_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: I
7assignvariableop_15_adam_base_model_8_dense_32_kernel_m:	C
5assignvariableop_16_adam_base_model_8_dense_32_bias_m:I
7assignvariableop_17_adam_base_model_8_dense_33_kernel_m: C
5assignvariableop_18_adam_base_model_8_dense_33_bias_m: I
7assignvariableop_19_adam_base_model_8_dense_34_kernel_m: C
5assignvariableop_20_adam_base_model_8_dense_34_bias_m:I
7assignvariableop_21_adam_base_model_8_dense_35_kernel_m:C
5assignvariableop_22_adam_base_model_8_dense_35_bias_m:I
7assignvariableop_23_adam_base_model_8_dense_32_kernel_v:	C
5assignvariableop_24_adam_base_model_8_dense_32_bias_v:I
7assignvariableop_25_adam_base_model_8_dense_33_kernel_v: C
5assignvariableop_26_adam_base_model_8_dense_33_bias_v: I
7assignvariableop_27_adam_base_model_8_dense_34_kernel_v: C
5assignvariableop_28_adam_base_model_8_dense_34_bias_v:I
7assignvariableop_29_adam_base_model_8_dense_35_kernel_v:C
5assignvariableop_30_adam_base_model_8_dense_35_bias_v:
identity_32ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ё
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*Д
valueъBч B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH∞
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ѕ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ц
_output_shapesГ
А::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOpAssignVariableOp-assignvariableop_base_model_8_dense_32_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_1AssignVariableOp-assignvariableop_1_base_model_8_dense_32_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_2AssignVariableOp/assignvariableop_2_base_model_8_dense_33_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_3AssignVariableOp-assignvariableop_3_base_model_8_dense_33_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_4AssignVariableOp/assignvariableop_4_base_model_8_dense_34_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_5AssignVariableOp-assignvariableop_5_base_model_8_dense_34_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_6AssignVariableOp/assignvariableop_6_base_model_8_dense_35_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_7AssignVariableOp-assignvariableop_7_base_model_8_dense_35_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_15AssignVariableOp7assignvariableop_15_adam_base_model_8_dense_32_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_16AssignVariableOp5assignvariableop_16_adam_base_model_8_dense_32_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adam_base_model_8_dense_33_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_18AssignVariableOp5assignvariableop_18_adam_base_model_8_dense_33_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_19AssignVariableOp7assignvariableop_19_adam_base_model_8_dense_34_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adam_base_model_8_dense_34_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adam_base_model_8_dense_35_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_22AssignVariableOp5assignvariableop_22_adam_base_model_8_dense_35_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_23AssignVariableOp7assignvariableop_23_adam_base_model_8_dense_32_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adam_base_model_8_dense_32_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_base_model_8_dense_33_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_base_model_8_dense_33_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_27AssignVariableOp7assignvariableop_27_adam_base_model_8_dense_34_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_base_model_8_dense_34_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_base_model_8_dense_35_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adam_base_model_8_dense_35_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 щ
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: ж
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ђ
serving_defaultЧ
;
input_10
serving_default_input_1:0€€€€€€€€€	<
output_10
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:бn
Ч
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1

	dense2


dense3
output_layer
	optimizer

signatures"
_tf_keras_model
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Њ
trace_0
trace_12З
/__inference_base_model_8_layer_call_fn_11150234
/__inference_base_model_8_layer_call_fn_11150369Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ztrace_0ztrace_1
ф
trace_0
trace_12љ
J__inference_base_model_8_layer_call_and_return_conditional_losses_11150400
J__inference_base_model_8_layer_call_and_return_conditional_losses_11150319Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ztrace_0ztrace_1
ќBЋ
#__inference__wrapped_model_11150140input_1"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ї
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ї
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ї
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ї
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
г
7iter

8beta_1

9beta_2
	:decay
;learning_ratem^m_m`mambmcmdmevfvgvhvivjvkvlvm"
	optimizer
,
<serving_default"
signature_map
.:,	2base_model_8/dense_32/kernel
(:&2base_model_8/dense_32/bias
.:, 2base_model_8/dense_33/kernel
(:& 2base_model_8/dense_33/bias
.:, 2base_model_8/dense_34/kernel
(:&2base_model_8/dense_34/bias
.:,2base_model_8/dense_35/kernel
(:&2base_model_8/dense_35/bias
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
/__inference_base_model_8_layer_call_fn_11150234input_1"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
гBа
/__inference_base_model_8_layer_call_fn_11150369inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
J__inference_base_model_8_layer_call_and_return_conditional_losses_11150400inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
€Bь
J__inference_base_model_8_layer_call_and_return_conditional_losses_11150319input_1"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
п
Ctrace_02“
+__inference_dense_32_layer_call_fn_11150409Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zCtrace_0
К
Dtrace_02н
F__inference_dense_32_layer_call_and_return_conditional_losses_11150420Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zDtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
п
Jtrace_02“
+__inference_dense_33_layer_call_fn_11150429Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zJtrace_0
К
Ktrace_02н
F__inference_dense_33_layer_call_and_return_conditional_losses_11150440Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zKtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
п
Qtrace_02“
+__inference_dense_34_layer_call_fn_11150449Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zQtrace_0
К
Rtrace_02н
F__inference_dense_34_layer_call_and_return_conditional_losses_11150460Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zRtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
п
Xtrace_02“
+__inference_dense_35_layer_call_fn_11150469Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zXtrace_0
К
Ytrace_02н
F__inference_dense_35_layer_call_and_return_conditional_losses_11150479Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zYtrace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ЌB 
&__inference_signature_wrapper_11150348input_1"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
N
Z	variables
[	keras_api
	\total
	]count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB№
+__inference_dense_32_layer_call_fn_11150409inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
F__inference_dense_32_layer_call_and_return_conditional_losses_11150420inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB№
+__inference_dense_33_layer_call_fn_11150429inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
F__inference_dense_33_layer_call_and_return_conditional_losses_11150440inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB№
+__inference_dense_34_layer_call_fn_11150449inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
F__inference_dense_34_layer_call_and_return_conditional_losses_11150460inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB№
+__inference_dense_35_layer_call_fn_11150469inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
F__inference_dense_35_layer_call_and_return_conditional_losses_11150479inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
\0
]1"
trackable_list_wrapper
-
Z	variables"
_generic_user_object
:  (2total
:  (2count
3:1	2#Adam/base_model_8/dense_32/kernel/m
-:+2!Adam/base_model_8/dense_32/bias/m
3:1 2#Adam/base_model_8/dense_33/kernel/m
-:+ 2!Adam/base_model_8/dense_33/bias/m
3:1 2#Adam/base_model_8/dense_34/kernel/m
-:+2!Adam/base_model_8/dense_34/bias/m
3:12#Adam/base_model_8/dense_35/kernel/m
-:+2!Adam/base_model_8/dense_35/bias/m
3:1	2#Adam/base_model_8/dense_32/kernel/v
-:+2!Adam/base_model_8/dense_32/bias/v
3:1 2#Adam/base_model_8/dense_33/kernel/v
-:+ 2!Adam/base_model_8/dense_33/bias/v
3:1 2#Adam/base_model_8/dense_34/kernel/v
-:+2!Adam/base_model_8/dense_34/bias/v
3:12#Adam/base_model_8/dense_35/kernel/v
-:+2!Adam/base_model_8/dense_35/bias/vШ
#__inference__wrapped_model_11150140q0Ґ-
&Ґ#
!К
input_1€€€€€€€€€	
™ "3™0
.
output_1"К
output_1€€€€€€€€€±
J__inference_base_model_8_layer_call_and_return_conditional_losses_11150319c0Ґ-
&Ґ#
!К
input_1€€€€€€€€€	
™ "%Ґ"
К
0€€€€€€€€€
Ъ ∞
J__inference_base_model_8_layer_call_and_return_conditional_losses_11150400b/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%Ґ"
К
0€€€€€€€€€
Ъ Й
/__inference_base_model_8_layer_call_fn_11150234V0Ґ-
&Ґ#
!К
input_1€€€€€€€€€	
™ "К€€€€€€€€€И
/__inference_base_model_8_layer_call_fn_11150369U/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "К€€€€€€€€€¶
F__inference_dense_32_layer_call_and_return_conditional_losses_11150420\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%Ґ"
К
0€€€€€€€€€
Ъ ~
+__inference_dense_32_layer_call_fn_11150409O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "К€€€€€€€€€¶
F__inference_dense_33_layer_call_and_return_conditional_losses_11150440\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ~
+__inference_dense_33_layer_call_fn_11150429O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€ ¶
F__inference_dense_34_layer_call_and_return_conditional_losses_11150460\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ~
+__inference_dense_34_layer_call_fn_11150449O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€¶
F__inference_dense_35_layer_call_and_return_conditional_losses_11150479\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ ~
+__inference_dense_35_layer_call_fn_11150469O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€¶
&__inference_signature_wrapper_11150348|;Ґ8
Ґ 
1™.
,
input_1!К
input_1€€€€€€€€€	"3™0
.
output_1"К
output_1€€€€€€€€€