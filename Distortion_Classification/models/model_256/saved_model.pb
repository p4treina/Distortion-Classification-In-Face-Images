??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.1-0-g85c8b2a817f8??

|
CONV1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameCONV1/kernel
u
 CONV1/kernel/Read/ReadVariableOpReadVariableOpCONV1/kernel*&
_output_shapes
:*
dtype0
l

CONV1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
CONV1/bias
e
CONV1/bias/Read/ReadVariableOpReadVariableOp
CONV1/bias*
_output_shapes
:*
dtype0
|
CONV2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameCONV2/kernel
u
 CONV2/kernel/Read/ReadVariableOpReadVariableOpCONV2/kernel*&
_output_shapes
: *
dtype0
l

CONV2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
CONV2/bias
e
CONV2/bias/Read/ReadVariableOpReadVariableOp
CONV2/bias*
_output_shapes
: *
dtype0
|
CONV3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameCONV3/kernel
u
 CONV3/kernel/Read/ReadVariableOpReadVariableOpCONV3/kernel*&
_output_shapes
: @*
dtype0
l

CONV3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
CONV3/bias
e
CONV3/bias/Read/ReadVariableOpReadVariableOp
CONV3/bias*
_output_shapes
:@*
dtype0
}
CONV4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*
shared_nameCONV4/kernel
v
 CONV4/kernel/Read/ReadVariableOpReadVariableOpCONV4/kernel*'
_output_shapes
:@?*
dtype0
m

CONV4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
CONV4/bias
f
CONV4/bias/Read/ReadVariableOpReadVariableOp
CONV4/bias*
_output_shapes	
:?*
dtype0
~
CONV5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameCONV5/kernel
w
 CONV5/kernel/Read/ReadVariableOpReadVariableOpCONV5/kernel*(
_output_shapes
:??*
dtype0
m

CONV5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
CONV5/bias
f
CONV5/bias/Read/ReadVariableOpReadVariableOp
CONV5/bias*
_output_shapes	
:?*
dtype0
r

FC1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_name
FC1/kernel
k
FC1/kernel/Read/ReadVariableOpReadVariableOp
FC1/kernel* 
_output_shapes
:
??*
dtype0
i
FC1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
FC1/bias
b
FC1/bias/Read/ReadVariableOpReadVariableOpFC1/bias*
_output_shapes	
:?*
dtype0
w
OUTPUT/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameOUTPUT/kernel
p
!OUTPUT/kernel/Read/ReadVariableOpReadVariableOpOUTPUT/kernel*
_output_shapes
:	?*
dtype0
n
OUTPUT/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameOUTPUT/bias
g
OUTPUT/bias/Read/ReadVariableOpReadVariableOpOUTPUT/bias*
_output_shapes
:*
dtype0
\
iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameiter
U
iter/Read/ReadVariableOpReadVariableOpiter*
_output_shapes
: *
dtype0	
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
CONV1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameCONV1/kernel/m
y
"CONV1/kernel/m/Read/ReadVariableOpReadVariableOpCONV1/kernel/m*&
_output_shapes
:*
dtype0
p
CONV1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameCONV1/bias/m
i
 CONV1/bias/m/Read/ReadVariableOpReadVariableOpCONV1/bias/m*
_output_shapes
:*
dtype0
?
CONV2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameCONV2/kernel/m
y
"CONV2/kernel/m/Read/ReadVariableOpReadVariableOpCONV2/kernel/m*&
_output_shapes
: *
dtype0
p
CONV2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameCONV2/bias/m
i
 CONV2/bias/m/Read/ReadVariableOpReadVariableOpCONV2/bias/m*
_output_shapes
: *
dtype0
?
CONV3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameCONV3/kernel/m
y
"CONV3/kernel/m/Read/ReadVariableOpReadVariableOpCONV3/kernel/m*&
_output_shapes
: @*
dtype0
p
CONV3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameCONV3/bias/m
i
 CONV3/bias/m/Read/ReadVariableOpReadVariableOpCONV3/bias/m*
_output_shapes
:@*
dtype0
?
CONV4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*
shared_nameCONV4/kernel/m
z
"CONV4/kernel/m/Read/ReadVariableOpReadVariableOpCONV4/kernel/m*'
_output_shapes
:@?*
dtype0
q
CONV4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameCONV4/bias/m
j
 CONV4/bias/m/Read/ReadVariableOpReadVariableOpCONV4/bias/m*
_output_shapes	
:?*
dtype0
?
CONV5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameCONV5/kernel/m
{
"CONV5/kernel/m/Read/ReadVariableOpReadVariableOpCONV5/kernel/m*(
_output_shapes
:??*
dtype0
q
CONV5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameCONV5/bias/m
j
 CONV5/bias/m/Read/ReadVariableOpReadVariableOpCONV5/bias/m*
_output_shapes	
:?*
dtype0
v
FC1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameFC1/kernel/m
o
 FC1/kernel/m/Read/ReadVariableOpReadVariableOpFC1/kernel/m* 
_output_shapes
:
??*
dtype0
m

FC1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
FC1/bias/m
f
FC1/bias/m/Read/ReadVariableOpReadVariableOp
FC1/bias/m*
_output_shapes	
:?*
dtype0
{
OUTPUT/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_nameOUTPUT/kernel/m
t
#OUTPUT/kernel/m/Read/ReadVariableOpReadVariableOpOUTPUT/kernel/m*
_output_shapes
:	?*
dtype0
r
OUTPUT/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameOUTPUT/bias/m
k
!OUTPUT/bias/m/Read/ReadVariableOpReadVariableOpOUTPUT/bias/m*
_output_shapes
:*
dtype0
?
CONV1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameCONV1/kernel/v
y
"CONV1/kernel/v/Read/ReadVariableOpReadVariableOpCONV1/kernel/v*&
_output_shapes
:*
dtype0
p
CONV1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameCONV1/bias/v
i
 CONV1/bias/v/Read/ReadVariableOpReadVariableOpCONV1/bias/v*
_output_shapes
:*
dtype0
?
CONV2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameCONV2/kernel/v
y
"CONV2/kernel/v/Read/ReadVariableOpReadVariableOpCONV2/kernel/v*&
_output_shapes
: *
dtype0
p
CONV2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameCONV2/bias/v
i
 CONV2/bias/v/Read/ReadVariableOpReadVariableOpCONV2/bias/v*
_output_shapes
: *
dtype0
?
CONV3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameCONV3/kernel/v
y
"CONV3/kernel/v/Read/ReadVariableOpReadVariableOpCONV3/kernel/v*&
_output_shapes
: @*
dtype0
p
CONV3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameCONV3/bias/v
i
 CONV3/bias/v/Read/ReadVariableOpReadVariableOpCONV3/bias/v*
_output_shapes
:@*
dtype0
?
CONV4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*
shared_nameCONV4/kernel/v
z
"CONV4/kernel/v/Read/ReadVariableOpReadVariableOpCONV4/kernel/v*'
_output_shapes
:@?*
dtype0
q
CONV4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameCONV4/bias/v
j
 CONV4/bias/v/Read/ReadVariableOpReadVariableOpCONV4/bias/v*
_output_shapes	
:?*
dtype0
?
CONV5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameCONV5/kernel/v
{
"CONV5/kernel/v/Read/ReadVariableOpReadVariableOpCONV5/kernel/v*(
_output_shapes
:??*
dtype0
q
CONV5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameCONV5/bias/v
j
 CONV5/bias/v/Read/ReadVariableOpReadVariableOpCONV5/bias/v*
_output_shapes	
:?*
dtype0
v
FC1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_nameFC1/kernel/v
o
 FC1/kernel/v/Read/ReadVariableOpReadVariableOpFC1/kernel/v* 
_output_shapes
:
??*
dtype0
m

FC1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
FC1/bias/v
f
FC1/bias/v/Read/ReadVariableOpReadVariableOp
FC1/bias/v*
_output_shapes	
:?*
dtype0
{
OUTPUT/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_nameOUTPUT/kernel/v
t
#OUTPUT/kernel/v/Read/ReadVariableOpReadVariableOpOUTPUT/kernel/v*
_output_shapes
:	?*
dtype0
r
OUTPUT/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameOUTPUT/bias/v
k
!OUTPUT/bias/v/Read/ReadVariableOpReadVariableOpOUTPUT/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?V
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?V
value?VB?V B?V
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
R
%regularization_losses
&trainable_variables
'	variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
R
/regularization_losses
0trainable_variables
1	variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
R
9regularization_losses
:trainable_variables
;	variables
<	keras_api
h

=kernel
>bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
R
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
R
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
h

Kkernel
Lbias
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
R
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
h

Ukernel
Vbias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
?
[iter

\beta_1

]beta_2
	^decay
_learning_ratem?m?m? m?)m?*m?3m?4m?=m?>m?Km?Lm?Um?Vm?v?v?v? v?)v?*v?3v?4v?=v?>v?Kv?Lv?Uv?Vv?
f
0
1
2
 3
)4
*5
36
47
=8
>9
K10
L11
U12
V13
 
f
0
1
2
 3
)4
*5
36
47
=8
>9
K10
L11
U12
V13
?
`metrics
anon_trainable_variables
trainable_variables
regularization_losses
blayer_metrics

clayers
	variables
dlayer_regularization_losses
 
XV
VARIABLE_VALUECONV1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
CONV1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
emetrics
fnon_trainable_variables
regularization_losses
trainable_variables
glayer_metrics

hlayers
	variables
ilayer_regularization_losses
 
 
 
?
jmetrics
knon_trainable_variables
regularization_losses
trainable_variables
llayer_metrics

mlayers
	variables
nlayer_regularization_losses
XV
VARIABLE_VALUECONV2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
CONV2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
?
ometrics
pnon_trainable_variables
!regularization_losses
"trainable_variables
qlayer_metrics

rlayers
#	variables
slayer_regularization_losses
 
 
 
?
tmetrics
unon_trainable_variables
%regularization_losses
&trainable_variables
vlayer_metrics

wlayers
'	variables
xlayer_regularization_losses
XV
VARIABLE_VALUECONV3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
CONV3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
?
ymetrics
znon_trainable_variables
+regularization_losses
,trainable_variables
{layer_metrics

|layers
-	variables
}layer_regularization_losses
 
 
 
?
~metrics
non_trainable_variables
/regularization_losses
0trainable_variables
?layer_metrics
?layers
1	variables
 ?layer_regularization_losses
XV
VARIABLE_VALUECONV4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
CONV4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
?
?metrics
?non_trainable_variables
5regularization_losses
6trainable_variables
?layer_metrics
?layers
7	variables
 ?layer_regularization_losses
 
 
 
?
?metrics
?non_trainable_variables
9regularization_losses
:trainable_variables
?layer_metrics
?layers
;	variables
 ?layer_regularization_losses
XV
VARIABLE_VALUECONV5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
CONV5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
?
?metrics
?non_trainable_variables
?regularization_losses
@trainable_variables
?layer_metrics
?layers
A	variables
 ?layer_regularization_losses
 
 
 
?
?metrics
?non_trainable_variables
Cregularization_losses
Dtrainable_variables
?layer_metrics
?layers
E	variables
 ?layer_regularization_losses
 
 
 
?
?metrics
?non_trainable_variables
Gregularization_losses
Htrainable_variables
?layer_metrics
?layers
I	variables
 ?layer_regularization_losses
VT
VARIABLE_VALUE
FC1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEFC1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

K0
L1

K0
L1
?
?metrics
?non_trainable_variables
Mregularization_losses
Ntrainable_variables
?layer_metrics
?layers
O	variables
 ?layer_regularization_losses
 
 
 
?
?metrics
?non_trainable_variables
Qregularization_losses
Rtrainable_variables
?layer_metrics
?layers
S	variables
 ?layer_regularization_losses
YW
VARIABLE_VALUEOUTPUT/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEOUTPUT/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

U0
V1

U0
V1
?
?metrics
?non_trainable_variables
Wregularization_losses
Xtrainable_variables
?layer_metrics
?layers
Y	variables
 ?layer_regularization_losses
CA
VARIABLE_VALUEiter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 
 
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
vt
VARIABLE_VALUECONV1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUECONV1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUECONV2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUECONV2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUECONV3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUECONV3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUECONV4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUECONV4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUECONV5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUECONV5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEFC1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE
FC1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEOUTPUT/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEOUTPUT/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUECONV1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUECONV1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUECONV2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUECONV2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUECONV3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUECONV3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUECONV4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUECONV4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUECONV5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUECONV5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEFC1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE
FC1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEOUTPUT/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEOUTPUT/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_CONV1_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_CONV1_inputCONV1/kernel
CONV1/biasCONV2/kernel
CONV2/biasCONV3/kernel
CONV3/biasCONV4/kernel
CONV4/biasCONV5/kernel
CONV5/bias
FC1/kernelFC1/biasOUTPUT/kernelOUTPUT/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_26773
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename CONV1/kernel/Read/ReadVariableOpCONV1/bias/Read/ReadVariableOp CONV2/kernel/Read/ReadVariableOpCONV2/bias/Read/ReadVariableOp CONV3/kernel/Read/ReadVariableOpCONV3/bias/Read/ReadVariableOp CONV4/kernel/Read/ReadVariableOpCONV4/bias/Read/ReadVariableOp CONV5/kernel/Read/ReadVariableOpCONV5/bias/Read/ReadVariableOpFC1/kernel/Read/ReadVariableOpFC1/bias/Read/ReadVariableOp!OUTPUT/kernel/Read/ReadVariableOpOUTPUT/bias/Read/ReadVariableOpiter/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"CONV1/kernel/m/Read/ReadVariableOp CONV1/bias/m/Read/ReadVariableOp"CONV2/kernel/m/Read/ReadVariableOp CONV2/bias/m/Read/ReadVariableOp"CONV3/kernel/m/Read/ReadVariableOp CONV3/bias/m/Read/ReadVariableOp"CONV4/kernel/m/Read/ReadVariableOp CONV4/bias/m/Read/ReadVariableOp"CONV5/kernel/m/Read/ReadVariableOp CONV5/bias/m/Read/ReadVariableOp FC1/kernel/m/Read/ReadVariableOpFC1/bias/m/Read/ReadVariableOp#OUTPUT/kernel/m/Read/ReadVariableOp!OUTPUT/bias/m/Read/ReadVariableOp"CONV1/kernel/v/Read/ReadVariableOp CONV1/bias/v/Read/ReadVariableOp"CONV2/kernel/v/Read/ReadVariableOp CONV2/bias/v/Read/ReadVariableOp"CONV3/kernel/v/Read/ReadVariableOp CONV3/bias/v/Read/ReadVariableOp"CONV4/kernel/v/Read/ReadVariableOp CONV4/bias/v/Read/ReadVariableOp"CONV5/kernel/v/Read/ReadVariableOp CONV5/bias/v/Read/ReadVariableOp FC1/kernel/v/Read/ReadVariableOpFC1/bias/v/Read/ReadVariableOp#OUTPUT/kernel/v/Read/ReadVariableOp!OUTPUT/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_27345
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameCONV1/kernel
CONV1/biasCONV2/kernel
CONV2/biasCONV3/kernel
CONV3/biasCONV4/kernel
CONV4/biasCONV5/kernel
CONV5/bias
FC1/kernelFC1/biasOUTPUT/kernelOUTPUT/biasiterbeta_1beta_2decaylearning_ratetotalcounttotal_1count_1CONV1/kernel/mCONV1/bias/mCONV2/kernel/mCONV2/bias/mCONV3/kernel/mCONV3/bias/mCONV4/kernel/mCONV4/bias/mCONV5/kernel/mCONV5/bias/mFC1/kernel/m
FC1/bias/mOUTPUT/kernel/mOUTPUT/bias/mCONV1/kernel/vCONV1/bias/vCONV2/kernel/vCONV2/bias/vCONV3/kernel/vCONV3/bias/vCONV4/kernel/vCONV4/bias/vCONV5/kernel/vCONV5/bias/vFC1/kernel/v
FC1/bias/vOUTPUT/kernel/vOUTPUT/bias/v*?
Tin8
624*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_27508??	
?
D
(__inference_DROPOUT5_layer_call_fn_27149

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_DROPOUT5_layer_call_and_return_conditional_losses_264842
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
@__inference_CONV4_layer_call_and_return_conditional_losses_27046

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
z
%__inference_CONV1_layer_call_fn_26995

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV1_layer_call_and_return_conditional_losses_262812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
D
(__inference_DROPOUT4_layer_call_fn_27102

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_DROPOUT4_layer_call_and_return_conditional_losses_264272
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?d
?
__inference__traced_save_27345
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop+
'savev2_conv3_kernel_read_readvariableop)
%savev2_conv3_bias_read_readvariableop+
'savev2_conv4_kernel_read_readvariableop)
%savev2_conv4_bias_read_readvariableop+
'savev2_conv5_kernel_read_readvariableop)
%savev2_conv5_bias_read_readvariableop)
%savev2_fc1_kernel_read_readvariableop'
#savev2_fc1_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop#
savev2_iter_read_readvariableop	%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_conv1_kernel_m_read_readvariableop+
'savev2_conv1_bias_m_read_readvariableop-
)savev2_conv2_kernel_m_read_readvariableop+
'savev2_conv2_bias_m_read_readvariableop-
)savev2_conv3_kernel_m_read_readvariableop+
'savev2_conv3_bias_m_read_readvariableop-
)savev2_conv4_kernel_m_read_readvariableop+
'savev2_conv4_bias_m_read_readvariableop-
)savev2_conv5_kernel_m_read_readvariableop+
'savev2_conv5_bias_m_read_readvariableop+
'savev2_fc1_kernel_m_read_readvariableop)
%savev2_fc1_bias_m_read_readvariableop.
*savev2_output_kernel_m_read_readvariableop,
(savev2_output_bias_m_read_readvariableop-
)savev2_conv1_kernel_v_read_readvariableop+
'savev2_conv1_bias_v_read_readvariableop-
)savev2_conv2_kernel_v_read_readvariableop+
'savev2_conv2_bias_v_read_readvariableop-
)savev2_conv3_kernel_v_read_readvariableop+
'savev2_conv3_bias_v_read_readvariableop-
)savev2_conv4_kernel_v_read_readvariableop+
'savev2_conv4_bias_v_read_readvariableop-
)savev2_conv5_kernel_v_read_readvariableop+
'savev2_conv5_bias_v_read_readvariableop+
'savev2_fc1_kernel_v_read_readvariableop)
%savev2_fc1_bias_v_read_readvariableop.
*savev2_output_kernel_v_read_readvariableop,
(savev2_output_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*?
value?B?4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop'savev2_conv3_kernel_read_readvariableop%savev2_conv3_bias_read_readvariableop'savev2_conv4_kernel_read_readvariableop%savev2_conv4_bias_read_readvariableop'savev2_conv5_kernel_read_readvariableop%savev2_conv5_bias_read_readvariableop%savev2_fc1_kernel_read_readvariableop#savev2_fc1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableopsavev2_iter_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_conv1_kernel_m_read_readvariableop'savev2_conv1_bias_m_read_readvariableop)savev2_conv2_kernel_m_read_readvariableop'savev2_conv2_bias_m_read_readvariableop)savev2_conv3_kernel_m_read_readvariableop'savev2_conv3_bias_m_read_readvariableop)savev2_conv4_kernel_m_read_readvariableop'savev2_conv4_bias_m_read_readvariableop)savev2_conv5_kernel_m_read_readvariableop'savev2_conv5_bias_m_read_readvariableop'savev2_fc1_kernel_m_read_readvariableop%savev2_fc1_bias_m_read_readvariableop*savev2_output_kernel_m_read_readvariableop(savev2_output_bias_m_read_readvariableop)savev2_conv1_kernel_v_read_readvariableop'savev2_conv1_bias_v_read_readvariableop)savev2_conv2_kernel_v_read_readvariableop'savev2_conv2_bias_v_read_readvariableop)savev2_conv3_kernel_v_read_readvariableop'savev2_conv3_bias_v_read_readvariableop)savev2_conv4_kernel_v_read_readvariableop'savev2_conv4_bias_v_read_readvariableop)savev2_conv5_kernel_v_read_readvariableop'savev2_conv5_bias_v_read_readvariableop'savev2_fc1_kernel_v_read_readvariableop%savev2_fc1_bias_v_read_readvariableop*savev2_output_kernel_v_read_readvariableop(savev2_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : @:@:@?:?:??:?:
??:?:	?:: : : : : : : : : ::: : : @:@:@?:?:??:?:
??:?:	?:::: : : @:@:@?:?:??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.	*
(
_output_shapes
:??:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:. *
(
_output_shapes
:??:!!

_output_shapes	
:?:&""
 
_output_shapes
:
??:!#

_output_shapes	
:?:%$!

_output_shapes
:	?: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
: : )

_output_shapes
: :,*(
&
_output_shapes
: @: +

_output_shapes
:@:-,)
'
_output_shapes
:@?:!-

_output_shapes	
:?:..*
(
_output_shapes
:??:!/

_output_shapes	
:?:&0"
 
_output_shapes
:
??:!1

_output_shapes	
:?:%2!

_output_shapes
:	?: 3

_output_shapes
::4

_output_shapes
: 
?
b
C__inference_DROPOUT4_layer_call_and_return_conditional_losses_27087

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
@__inference_CONV4_layer_call_and_return_conditional_losses_26365

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
C__inference_DROPOUT4_layer_call_and_return_conditional_losses_26427

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
{
&__inference_OUTPUT_layer_call_fn_27169

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_OUTPUT_layer_call_and_return_conditional_losses_265082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_MAXPOOL2_layer_call_and_return_conditional_losses_26223

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
z
%__inference_CONV4_layer_call_fn_27055

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV4_layer_call_and_return_conditional_losses_263652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
@__inference_CONV5_layer_call_and_return_conditional_losses_27066

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
>__inference_FC1_layer_call_and_return_conditional_losses_27113

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
A__inference_OUTPUT_layer_call_and_return_conditional_losses_26508

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
z
%__inference_CONV3_layer_call_fn_27035

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV3_layer_call_and_return_conditional_losses_263372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????>> ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????>> 
 
_user_specified_nameinputs
?
a
C__inference_DROPOUT5_layer_call_and_return_conditional_losses_27139

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
@__inference_CONV1_layer_call_and_return_conditional_losses_26281

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
@__inference_CONV1_layer_call_and_return_conditional_losses_26986

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
D
(__inference_MAXPOOL3_layer_call_fn_26241

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL3_layer_call_and_return_conditional_losses_262352
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
z
%__inference_CONV2_layer_call_fn_27015

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????}} *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV2_layer_call_and_return_conditional_losses_263092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????}} 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
,__inference_sequential_1_layer_call_fn_26942

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_266202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
>__inference_FC1_layer_call_and_return_conditional_losses_26451

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
,__inference_sequential_1_layer_call_fn_26651
conv1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_266202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameCONV1_input
?
z
%__inference_CONV5_layer_call_fn_27075

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV5_layer_call_and_return_conditional_losses_263932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
C__inference_DROPOUT5_layer_call_and_return_conditional_losses_26484

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
C__inference_DROPOUT5_layer_call_and_return_conditional_losses_27134

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_27508
file_prefix!
assignvariableop_conv1_kernel!
assignvariableop_1_conv1_bias#
assignvariableop_2_conv2_kernel!
assignvariableop_3_conv2_bias#
assignvariableop_4_conv3_kernel!
assignvariableop_5_conv3_bias#
assignvariableop_6_conv4_kernel!
assignvariableop_7_conv4_bias#
assignvariableop_8_conv5_kernel!
assignvariableop_9_conv5_bias"
assignvariableop_10_fc1_kernel 
assignvariableop_11_fc1_bias%
!assignvariableop_12_output_kernel#
assignvariableop_13_output_bias
assignvariableop_14_iter
assignvariableop_15_beta_1
assignvariableop_16_beta_2
assignvariableop_17_decay%
!assignvariableop_18_learning_rate
assignvariableop_19_total
assignvariableop_20_count
assignvariableop_21_total_1
assignvariableop_22_count_1&
"assignvariableop_23_conv1_kernel_m$
 assignvariableop_24_conv1_bias_m&
"assignvariableop_25_conv2_kernel_m$
 assignvariableop_26_conv2_bias_m&
"assignvariableop_27_conv3_kernel_m$
 assignvariableop_28_conv3_bias_m&
"assignvariableop_29_conv4_kernel_m$
 assignvariableop_30_conv4_bias_m&
"assignvariableop_31_conv5_kernel_m$
 assignvariableop_32_conv5_bias_m$
 assignvariableop_33_fc1_kernel_m"
assignvariableop_34_fc1_bias_m'
#assignvariableop_35_output_kernel_m%
!assignvariableop_36_output_bias_m&
"assignvariableop_37_conv1_kernel_v$
 assignvariableop_38_conv1_bias_v&
"assignvariableop_39_conv2_kernel_v$
 assignvariableop_40_conv2_bias_v&
"assignvariableop_41_conv3_kernel_v$
 assignvariableop_42_conv3_bias_v&
"assignvariableop_43_conv4_kernel_v$
 assignvariableop_44_conv4_bias_v&
"assignvariableop_45_conv5_kernel_v$
 assignvariableop_46_conv5_bias_v$
 assignvariableop_47_fc1_kernel_v"
assignvariableop_48_fc1_bias_v'
#assignvariableop_49_output_kernel_v%
!assignvariableop_50_output_bias_v
identity_52??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*?
value?B?4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_conv3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_conv5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_fc1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_fc1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_output_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_output_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp!assignvariableop_18_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp assignvariableop_24_conv1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp assignvariableop_26_conv2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv3_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp assignvariableop_28_conv3_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv4_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp assignvariableop_30_conv4_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv5_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp assignvariableop_32_conv5_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp assignvariableop_33_fc1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_fc1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp#assignvariableop_35_output_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp!assignvariableop_36_output_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp assignvariableop_38_conv1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp"assignvariableop_39_conv2_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp assignvariableop_40_conv2_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp"assignvariableop_41_conv3_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp assignvariableop_42_conv3_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp"assignvariableop_43_conv4_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp assignvariableop_44_conv4_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp"assignvariableop_45_conv5_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp assignvariableop_46_conv5_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp assignvariableop_47_fc1_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_fc1_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp#assignvariableop_49_output_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp!assignvariableop_50_output_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_509
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_51?	
Identity_52IdentityIdentity_51:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_52"#
identity_52Identity_52:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
_
C__inference_MAXPOOL3_layer_call_and_return_conditional_losses_26235

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
D
(__inference_MAXPOOL2_layer_call_fn_26229

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL2_layer_call_and_return_conditional_losses_262232
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
,__inference_sequential_1_layer_call_fn_26730
conv1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_266992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameCONV1_input
?8
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26571
conv1_input
conv1_26528
conv1_26530
conv2_26534
conv2_26536
conv3_26540
conv3_26542
conv4_26546
conv4_26548
conv5_26552
conv5_26554
	fc1_26559
	fc1_26561
output_26565
output_26567
identity??CONV1/StatefulPartitionedCall?CONV2/StatefulPartitionedCall?CONV3/StatefulPartitionedCall?CONV4/StatefulPartitionedCall?CONV5/StatefulPartitionedCall?FC1/StatefulPartitionedCall?OUTPUT/StatefulPartitionedCall?
CONV1/StatefulPartitionedCallStatefulPartitionedCallconv1_inputconv1_26528conv1_26530*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV1_layer_call_and_return_conditional_losses_262812
CONV1/StatefulPartitionedCall?
MAXPOOL1/PartitionedCallPartitionedCall&CONV1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL1_layer_call_and_return_conditional_losses_262112
MAXPOOL1/PartitionedCall?
CONV2/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL1/PartitionedCall:output:0conv2_26534conv2_26536*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????}} *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV2_layer_call_and_return_conditional_losses_263092
CONV2/StatefulPartitionedCall?
MAXPOOL2/PartitionedCallPartitionedCall&CONV2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>> * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL2_layer_call_and_return_conditional_losses_262232
MAXPOOL2/PartitionedCall?
CONV3/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL2/PartitionedCall:output:0conv3_26540conv3_26542*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV3_layer_call_and_return_conditional_losses_263372
CONV3/StatefulPartitionedCall?
MAXPOOL3/PartitionedCallPartitionedCall&CONV3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL3_layer_call_and_return_conditional_losses_262352
MAXPOOL3/PartitionedCall?
CONV4/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL3/PartitionedCall:output:0conv4_26546conv4_26548*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV4_layer_call_and_return_conditional_losses_263652
CONV4/StatefulPartitionedCall?
MAXPOOL4/PartitionedCallPartitionedCall&CONV4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL4_layer_call_and_return_conditional_losses_262472
MAXPOOL4/PartitionedCall?
CONV5/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL4/PartitionedCall:output:0conv5_26552conv5_26554*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV5_layer_call_and_return_conditional_losses_263932
CONV5/StatefulPartitionedCall?
GLOBAL_MAXPOOL/PartitionedCallPartitionedCall&CONV5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_GLOBAL_MAXPOOL_layer_call_and_return_conditional_losses_262602 
GLOBAL_MAXPOOL/PartitionedCall?
DROPOUT4/PartitionedCallPartitionedCall'GLOBAL_MAXPOOL/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_DROPOUT4_layer_call_and_return_conditional_losses_264272
DROPOUT4/PartitionedCall?
FC1/StatefulPartitionedCallStatefulPartitionedCall!DROPOUT4/PartitionedCall:output:0	fc1_26559	fc1_26561*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_FC1_layer_call_and_return_conditional_losses_264512
FC1/StatefulPartitionedCall?
DROPOUT5/PartitionedCallPartitionedCall$FC1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_DROPOUT5_layer_call_and_return_conditional_losses_264842
DROPOUT5/PartitionedCall?
OUTPUT/StatefulPartitionedCallStatefulPartitionedCall!DROPOUT5/PartitionedCall:output:0output_26565output_26567*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_OUTPUT_layer_call_and_return_conditional_losses_265082 
OUTPUT/StatefulPartitionedCall?
IdentityIdentity'OUTPUT/StatefulPartitionedCall:output:0^CONV1/StatefulPartitionedCall^CONV2/StatefulPartitionedCall^CONV3/StatefulPartitionedCall^CONV4/StatefulPartitionedCall^CONV5/StatefulPartitionedCall^FC1/StatefulPartitionedCall^OUTPUT/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2>
CONV1/StatefulPartitionedCallCONV1/StatefulPartitionedCall2>
CONV2/StatefulPartitionedCallCONV2/StatefulPartitionedCall2>
CONV3/StatefulPartitionedCallCONV3/StatefulPartitionedCall2>
CONV4/StatefulPartitionedCallCONV4/StatefulPartitionedCall2>
CONV5/StatefulPartitionedCallCONV5/StatefulPartitionedCall2:
FC1/StatefulPartitionedCallFC1/StatefulPartitionedCall2@
OUTPUT/StatefulPartitionedCallOUTPUT/StatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameCONV1_input
?`
?
 __inference__wrapped_model_26205
conv1_input5
1sequential_1_conv1_conv2d_readvariableop_resource6
2sequential_1_conv1_biasadd_readvariableop_resource5
1sequential_1_conv2_conv2d_readvariableop_resource6
2sequential_1_conv2_biasadd_readvariableop_resource5
1sequential_1_conv3_conv2d_readvariableop_resource6
2sequential_1_conv3_biasadd_readvariableop_resource5
1sequential_1_conv4_conv2d_readvariableop_resource6
2sequential_1_conv4_biasadd_readvariableop_resource5
1sequential_1_conv5_conv2d_readvariableop_resource6
2sequential_1_conv5_biasadd_readvariableop_resource3
/sequential_1_fc1_matmul_readvariableop_resource4
0sequential_1_fc1_biasadd_readvariableop_resource6
2sequential_1_output_matmul_readvariableop_resource7
3sequential_1_output_biasadd_readvariableop_resource
identity??)sequential_1/CONV1/BiasAdd/ReadVariableOp?(sequential_1/CONV1/Conv2D/ReadVariableOp?)sequential_1/CONV2/BiasAdd/ReadVariableOp?(sequential_1/CONV2/Conv2D/ReadVariableOp?)sequential_1/CONV3/BiasAdd/ReadVariableOp?(sequential_1/CONV3/Conv2D/ReadVariableOp?)sequential_1/CONV4/BiasAdd/ReadVariableOp?(sequential_1/CONV4/Conv2D/ReadVariableOp?)sequential_1/CONV5/BiasAdd/ReadVariableOp?(sequential_1/CONV5/Conv2D/ReadVariableOp?'sequential_1/FC1/BiasAdd/ReadVariableOp?&sequential_1/FC1/MatMul/ReadVariableOp?*sequential_1/OUTPUT/BiasAdd/ReadVariableOp?)sequential_1/OUTPUT/MatMul/ReadVariableOp?
(sequential_1/CONV1/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(sequential_1/CONV1/Conv2D/ReadVariableOp?
sequential_1/CONV1/Conv2DConv2Dconv1_input0sequential_1/CONV1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
sequential_1/CONV1/Conv2D?
)sequential_1/CONV1/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_1/CONV1/BiasAdd/ReadVariableOp?
sequential_1/CONV1/BiasAddBiasAdd"sequential_1/CONV1/Conv2D:output:01sequential_1/CONV1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
sequential_1/CONV1/BiasAdd?
sequential_1/CONV1/ReluRelu#sequential_1/CONV1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
sequential_1/CONV1/Relu?
sequential_1/MAXPOOL1/MaxPoolMaxPool%sequential_1/CONV1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
sequential_1/MAXPOOL1/MaxPool?
(sequential_1/CONV2/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(sequential_1/CONV2/Conv2D/ReadVariableOp?
sequential_1/CONV2/Conv2DConv2D&sequential_1/MAXPOOL1/MaxPool:output:00sequential_1/CONV2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????}} *
paddingVALID*
strides
2
sequential_1/CONV2/Conv2D?
)sequential_1/CONV2/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)sequential_1/CONV2/BiasAdd/ReadVariableOp?
sequential_1/CONV2/BiasAddBiasAdd"sequential_1/CONV2/Conv2D:output:01sequential_1/CONV2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????}} 2
sequential_1/CONV2/BiasAdd?
sequential_1/CONV2/ReluRelu#sequential_1/CONV2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}} 2
sequential_1/CONV2/Relu?
sequential_1/MAXPOOL2/MaxPoolMaxPool%sequential_1/CONV2/Relu:activations:0*/
_output_shapes
:?????????>> *
ksize
*
paddingVALID*
strides
2
sequential_1/MAXPOOL2/MaxPool?
(sequential_1/CONV3/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02*
(sequential_1/CONV3/Conv2D/ReadVariableOp?
sequential_1/CONV3/Conv2DConv2D&sequential_1/MAXPOOL2/MaxPool:output:00sequential_1/CONV3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@*
paddingVALID*
strides
2
sequential_1/CONV3/Conv2D?
)sequential_1/CONV3/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)sequential_1/CONV3/BiasAdd/ReadVariableOp?
sequential_1/CONV3/BiasAddBiasAdd"sequential_1/CONV3/Conv2D:output:01sequential_1/CONV3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@2
sequential_1/CONV3/BiasAdd?
sequential_1/CONV3/ReluRelu#sequential_1/CONV3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<<@2
sequential_1/CONV3/Relu?
sequential_1/MAXPOOL3/MaxPoolMaxPool%sequential_1/CONV3/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
sequential_1/MAXPOOL3/MaxPool?
(sequential_1/CONV4/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02*
(sequential_1/CONV4/Conv2D/ReadVariableOp?
sequential_1/CONV4/Conv2DConv2D&sequential_1/MAXPOOL3/MaxPool:output:00sequential_1/CONV4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential_1/CONV4/Conv2D?
)sequential_1/CONV4/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential_1/CONV4/BiasAdd/ReadVariableOp?
sequential_1/CONV4/BiasAddBiasAdd"sequential_1/CONV4/Conv2D:output:01sequential_1/CONV4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential_1/CONV4/BiasAdd?
sequential_1/CONV4/ReluRelu#sequential_1/CONV4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential_1/CONV4/Relu?
sequential_1/MAXPOOL4/MaxPoolMaxPool%sequential_1/CONV4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
sequential_1/MAXPOOL4/MaxPool?
(sequential_1/CONV5/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(sequential_1/CONV5/Conv2D/ReadVariableOp?
sequential_1/CONV5/Conv2DConv2D&sequential_1/MAXPOOL4/MaxPool:output:00sequential_1/CONV5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
sequential_1/CONV5/Conv2D?
)sequential_1/CONV5/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential_1/CONV5/BiasAdd/ReadVariableOp?
sequential_1/CONV5/BiasAddBiasAdd"sequential_1/CONV5/Conv2D:output:01sequential_1/CONV5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
sequential_1/CONV5/BiasAdd?
sequential_1/CONV5/ReluRelu#sequential_1/CONV5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
sequential_1/CONV5/Relu?
1sequential_1/GLOBAL_MAXPOOL/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1sequential_1/GLOBAL_MAXPOOL/Max/reduction_indices?
sequential_1/GLOBAL_MAXPOOL/MaxMax%sequential_1/CONV5/Relu:activations:0:sequential_1/GLOBAL_MAXPOOL/Max/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2!
sequential_1/GLOBAL_MAXPOOL/Max?
sequential_1/DROPOUT4/IdentityIdentity(sequential_1/GLOBAL_MAXPOOL/Max:output:0*
T0*(
_output_shapes
:??????????2 
sequential_1/DROPOUT4/Identity?
&sequential_1/FC1/MatMul/ReadVariableOpReadVariableOp/sequential_1_fc1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&sequential_1/FC1/MatMul/ReadVariableOp?
sequential_1/FC1/MatMulMatMul'sequential_1/DROPOUT4/Identity:output:0.sequential_1/FC1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/FC1/MatMul?
'sequential_1/FC1/BiasAdd/ReadVariableOpReadVariableOp0sequential_1_fc1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'sequential_1/FC1/BiasAdd/ReadVariableOp?
sequential_1/FC1/BiasAddBiasAdd!sequential_1/FC1/MatMul:product:0/sequential_1/FC1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_1/FC1/BiasAdd?
sequential_1/FC1/ReluRelu!sequential_1/FC1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_1/FC1/Relu?
sequential_1/DROPOUT5/IdentityIdentity#sequential_1/FC1/Relu:activations:0*
T0*(
_output_shapes
:??????????2 
sequential_1/DROPOUT5/Identity?
)sequential_1/OUTPUT/MatMul/ReadVariableOpReadVariableOp2sequential_1_output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)sequential_1/OUTPUT/MatMul/ReadVariableOp?
sequential_1/OUTPUT/MatMulMatMul'sequential_1/DROPOUT5/Identity:output:01sequential_1/OUTPUT/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_1/OUTPUT/MatMul?
*sequential_1/OUTPUT/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_1/OUTPUT/BiasAdd/ReadVariableOp?
sequential_1/OUTPUT/BiasAddBiasAdd$sequential_1/OUTPUT/MatMul:product:02sequential_1/OUTPUT/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_1/OUTPUT/BiasAdd?
sequential_1/OUTPUT/SoftmaxSoftmax$sequential_1/OUTPUT/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_1/OUTPUT/Softmax?
IdentityIdentity%sequential_1/OUTPUT/Softmax:softmax:0*^sequential_1/CONV1/BiasAdd/ReadVariableOp)^sequential_1/CONV1/Conv2D/ReadVariableOp*^sequential_1/CONV2/BiasAdd/ReadVariableOp)^sequential_1/CONV2/Conv2D/ReadVariableOp*^sequential_1/CONV3/BiasAdd/ReadVariableOp)^sequential_1/CONV3/Conv2D/ReadVariableOp*^sequential_1/CONV4/BiasAdd/ReadVariableOp)^sequential_1/CONV4/Conv2D/ReadVariableOp*^sequential_1/CONV5/BiasAdd/ReadVariableOp)^sequential_1/CONV5/Conv2D/ReadVariableOp(^sequential_1/FC1/BiasAdd/ReadVariableOp'^sequential_1/FC1/MatMul/ReadVariableOp+^sequential_1/OUTPUT/BiasAdd/ReadVariableOp*^sequential_1/OUTPUT/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2V
)sequential_1/CONV1/BiasAdd/ReadVariableOp)sequential_1/CONV1/BiasAdd/ReadVariableOp2T
(sequential_1/CONV1/Conv2D/ReadVariableOp(sequential_1/CONV1/Conv2D/ReadVariableOp2V
)sequential_1/CONV2/BiasAdd/ReadVariableOp)sequential_1/CONV2/BiasAdd/ReadVariableOp2T
(sequential_1/CONV2/Conv2D/ReadVariableOp(sequential_1/CONV2/Conv2D/ReadVariableOp2V
)sequential_1/CONV3/BiasAdd/ReadVariableOp)sequential_1/CONV3/BiasAdd/ReadVariableOp2T
(sequential_1/CONV3/Conv2D/ReadVariableOp(sequential_1/CONV3/Conv2D/ReadVariableOp2V
)sequential_1/CONV4/BiasAdd/ReadVariableOp)sequential_1/CONV4/BiasAdd/ReadVariableOp2T
(sequential_1/CONV4/Conv2D/ReadVariableOp(sequential_1/CONV4/Conv2D/ReadVariableOp2V
)sequential_1/CONV5/BiasAdd/ReadVariableOp)sequential_1/CONV5/BiasAdd/ReadVariableOp2T
(sequential_1/CONV5/Conv2D/ReadVariableOp(sequential_1/CONV5/Conv2D/ReadVariableOp2R
'sequential_1/FC1/BiasAdd/ReadVariableOp'sequential_1/FC1/BiasAdd/ReadVariableOp2P
&sequential_1/FC1/MatMul/ReadVariableOp&sequential_1/FC1/MatMul/ReadVariableOp2X
*sequential_1/OUTPUT/BiasAdd/ReadVariableOp*sequential_1/OUTPUT/BiasAdd/ReadVariableOp2V
)sequential_1/OUTPUT/MatMul/ReadVariableOp)sequential_1/OUTPUT/MatMul/ReadVariableOp:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameCONV1_input
?

?
@__inference_CONV2_layer_call_and_return_conditional_losses_27006

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????}} *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????}} 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????}} 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????}} 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?;
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26525
conv1_input
conv1_26292
conv1_26294
conv2_26320
conv2_26322
conv3_26348
conv3_26350
conv4_26376
conv4_26378
conv5_26404
conv5_26406
	fc1_26462
	fc1_26464
output_26519
output_26521
identity??CONV1/StatefulPartitionedCall?CONV2/StatefulPartitionedCall?CONV3/StatefulPartitionedCall?CONV4/StatefulPartitionedCall?CONV5/StatefulPartitionedCall? DROPOUT4/StatefulPartitionedCall? DROPOUT5/StatefulPartitionedCall?FC1/StatefulPartitionedCall?OUTPUT/StatefulPartitionedCall?
CONV1/StatefulPartitionedCallStatefulPartitionedCallconv1_inputconv1_26292conv1_26294*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV1_layer_call_and_return_conditional_losses_262812
CONV1/StatefulPartitionedCall?
MAXPOOL1/PartitionedCallPartitionedCall&CONV1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL1_layer_call_and_return_conditional_losses_262112
MAXPOOL1/PartitionedCall?
CONV2/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL1/PartitionedCall:output:0conv2_26320conv2_26322*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????}} *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV2_layer_call_and_return_conditional_losses_263092
CONV2/StatefulPartitionedCall?
MAXPOOL2/PartitionedCallPartitionedCall&CONV2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>> * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL2_layer_call_and_return_conditional_losses_262232
MAXPOOL2/PartitionedCall?
CONV3/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL2/PartitionedCall:output:0conv3_26348conv3_26350*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV3_layer_call_and_return_conditional_losses_263372
CONV3/StatefulPartitionedCall?
MAXPOOL3/PartitionedCallPartitionedCall&CONV3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL3_layer_call_and_return_conditional_losses_262352
MAXPOOL3/PartitionedCall?
CONV4/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL3/PartitionedCall:output:0conv4_26376conv4_26378*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV4_layer_call_and_return_conditional_losses_263652
CONV4/StatefulPartitionedCall?
MAXPOOL4/PartitionedCallPartitionedCall&CONV4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL4_layer_call_and_return_conditional_losses_262472
MAXPOOL4/PartitionedCall?
CONV5/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL4/PartitionedCall:output:0conv5_26404conv5_26406*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV5_layer_call_and_return_conditional_losses_263932
CONV5/StatefulPartitionedCall?
GLOBAL_MAXPOOL/PartitionedCallPartitionedCall&CONV5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_GLOBAL_MAXPOOL_layer_call_and_return_conditional_losses_262602 
GLOBAL_MAXPOOL/PartitionedCall?
 DROPOUT4/StatefulPartitionedCallStatefulPartitionedCall'GLOBAL_MAXPOOL/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_DROPOUT4_layer_call_and_return_conditional_losses_264222"
 DROPOUT4/StatefulPartitionedCall?
FC1/StatefulPartitionedCallStatefulPartitionedCall)DROPOUT4/StatefulPartitionedCall:output:0	fc1_26462	fc1_26464*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_FC1_layer_call_and_return_conditional_losses_264512
FC1/StatefulPartitionedCall?
 DROPOUT5/StatefulPartitionedCallStatefulPartitionedCall$FC1/StatefulPartitionedCall:output:0!^DROPOUT4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_DROPOUT5_layer_call_and_return_conditional_losses_264792"
 DROPOUT5/StatefulPartitionedCall?
OUTPUT/StatefulPartitionedCallStatefulPartitionedCall)DROPOUT5/StatefulPartitionedCall:output:0output_26519output_26521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_OUTPUT_layer_call_and_return_conditional_losses_265082 
OUTPUT/StatefulPartitionedCall?
IdentityIdentity'OUTPUT/StatefulPartitionedCall:output:0^CONV1/StatefulPartitionedCall^CONV2/StatefulPartitionedCall^CONV3/StatefulPartitionedCall^CONV4/StatefulPartitionedCall^CONV5/StatefulPartitionedCall!^DROPOUT4/StatefulPartitionedCall!^DROPOUT5/StatefulPartitionedCall^FC1/StatefulPartitionedCall^OUTPUT/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2>
CONV1/StatefulPartitionedCallCONV1/StatefulPartitionedCall2>
CONV2/StatefulPartitionedCallCONV2/StatefulPartitionedCall2>
CONV3/StatefulPartitionedCallCONV3/StatefulPartitionedCall2>
CONV4/StatefulPartitionedCallCONV4/StatefulPartitionedCall2>
CONV5/StatefulPartitionedCallCONV5/StatefulPartitionedCall2D
 DROPOUT4/StatefulPartitionedCall DROPOUT4/StatefulPartitionedCall2D
 DROPOUT5/StatefulPartitionedCall DROPOUT5/StatefulPartitionedCall2:
FC1/StatefulPartitionedCallFC1/StatefulPartitionedCall2@
OUTPUT/StatefulPartitionedCallOUTPUT/StatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameCONV1_input
?
a
(__inference_DROPOUT5_layer_call_fn_27144

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_DROPOUT5_layer_call_and_return_conditional_losses_264792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_GLOBAL_MAXPOOL_layer_call_and_return_conditional_losses_26260

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?K
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26909

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource(
$conv5_conv2d_readvariableop_resource)
%conv5_biasadd_readvariableop_resource&
"fc1_matmul_readvariableop_resource'
#fc1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??CONV1/BiasAdd/ReadVariableOp?CONV1/Conv2D/ReadVariableOp?CONV2/BiasAdd/ReadVariableOp?CONV2/Conv2D/ReadVariableOp?CONV3/BiasAdd/ReadVariableOp?CONV3/Conv2D/ReadVariableOp?CONV4/BiasAdd/ReadVariableOp?CONV4/Conv2D/ReadVariableOp?CONV5/BiasAdd/ReadVariableOp?CONV5/Conv2D/ReadVariableOp?FC1/BiasAdd/ReadVariableOp?FC1/MatMul/ReadVariableOp?OUTPUT/BiasAdd/ReadVariableOp?OUTPUT/MatMul/ReadVariableOp?
CONV1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
CONV1/Conv2D/ReadVariableOp?
CONV1/Conv2DConv2Dinputs#CONV1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
CONV1/Conv2D?
CONV1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
CONV1/BiasAdd/ReadVariableOp?
CONV1/BiasAddBiasAddCONV1/Conv2D:output:0$CONV1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
CONV1/BiasAddt

CONV1/ReluReluCONV1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2

CONV1/Relu?
MAXPOOL1/MaxPoolMaxPoolCONV1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
MAXPOOL1/MaxPool?
CONV2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
CONV2/Conv2D/ReadVariableOp?
CONV2/Conv2DConv2DMAXPOOL1/MaxPool:output:0#CONV2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????}} *
paddingVALID*
strides
2
CONV2/Conv2D?
CONV2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
CONV2/BiasAdd/ReadVariableOp?
CONV2/BiasAddBiasAddCONV2/Conv2D:output:0$CONV2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????}} 2
CONV2/BiasAddr

CONV2/ReluReluCONV2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}} 2

CONV2/Relu?
MAXPOOL2/MaxPoolMaxPoolCONV2/Relu:activations:0*/
_output_shapes
:?????????>> *
ksize
*
paddingVALID*
strides
2
MAXPOOL2/MaxPool?
CONV3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
CONV3/Conv2D/ReadVariableOp?
CONV3/Conv2DConv2DMAXPOOL2/MaxPool:output:0#CONV3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@*
paddingVALID*
strides
2
CONV3/Conv2D?
CONV3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
CONV3/BiasAdd/ReadVariableOp?
CONV3/BiasAddBiasAddCONV3/Conv2D:output:0$CONV3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@2
CONV3/BiasAddr

CONV3/ReluReluCONV3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<<@2

CONV3/Relu?
MAXPOOL3/MaxPoolMaxPoolCONV3/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
MAXPOOL3/MaxPool?
CONV4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
CONV4/Conv2D/ReadVariableOp?
CONV4/Conv2DConv2DMAXPOOL3/MaxPool:output:0#CONV4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
CONV4/Conv2D?
CONV4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
CONV4/BiasAdd/ReadVariableOp?
CONV4/BiasAddBiasAddCONV4/Conv2D:output:0$CONV4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
CONV4/BiasAdds

CONV4/ReluReluCONV4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2

CONV4/Relu?
MAXPOOL4/MaxPoolMaxPoolCONV4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
MAXPOOL4/MaxPool?
CONV5/Conv2D/ReadVariableOpReadVariableOp$conv5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
CONV5/Conv2D/ReadVariableOp?
CONV5/Conv2DConv2DMAXPOOL4/MaxPool:output:0#CONV5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
CONV5/Conv2D?
CONV5/BiasAdd/ReadVariableOpReadVariableOp%conv5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
CONV5/BiasAdd/ReadVariableOp?
CONV5/BiasAddBiasAddCONV5/Conv2D:output:0$CONV5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
CONV5/BiasAdds

CONV5/ReluReluCONV5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2

CONV5/Relu?
$GLOBAL_MAXPOOL/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2&
$GLOBAL_MAXPOOL/Max/reduction_indices?
GLOBAL_MAXPOOL/MaxMaxCONV5/Relu:activations:0-GLOBAL_MAXPOOL/Max/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
GLOBAL_MAXPOOL/Max?
DROPOUT4/IdentityIdentityGLOBAL_MAXPOOL/Max:output:0*
T0*(
_output_shapes
:??????????2
DROPOUT4/Identity?
FC1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
FC1/MatMul/ReadVariableOp?

FC1/MatMulMatMulDROPOUT4/Identity:output:0!FC1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

FC1/MatMul?
FC1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
FC1/BiasAdd/ReadVariableOp?
FC1/BiasAddBiasAddFC1/MatMul:product:0"FC1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
FC1/BiasAdde
FC1/ReluReluFC1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

FC1/Relu}
DROPOUT5/IdentityIdentityFC1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
DROPOUT5/Identity?
OUTPUT/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
OUTPUT/MatMul/ReadVariableOp?
OUTPUT/MatMulMatMulDROPOUT5/Identity:output:0$OUTPUT/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
OUTPUT/MatMul?
OUTPUT/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
OUTPUT/BiasAdd/ReadVariableOp?
OUTPUT/BiasAddBiasAddOUTPUT/MatMul:product:0%OUTPUT/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
OUTPUT/BiasAddv
OUTPUT/SoftmaxSoftmaxOUTPUT/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
OUTPUT/Softmax?
IdentityIdentityOUTPUT/Softmax:softmax:0^CONV1/BiasAdd/ReadVariableOp^CONV1/Conv2D/ReadVariableOp^CONV2/BiasAdd/ReadVariableOp^CONV2/Conv2D/ReadVariableOp^CONV3/BiasAdd/ReadVariableOp^CONV3/Conv2D/ReadVariableOp^CONV4/BiasAdd/ReadVariableOp^CONV4/Conv2D/ReadVariableOp^CONV5/BiasAdd/ReadVariableOp^CONV5/Conv2D/ReadVariableOp^FC1/BiasAdd/ReadVariableOp^FC1/MatMul/ReadVariableOp^OUTPUT/BiasAdd/ReadVariableOp^OUTPUT/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2<
CONV1/BiasAdd/ReadVariableOpCONV1/BiasAdd/ReadVariableOp2:
CONV1/Conv2D/ReadVariableOpCONV1/Conv2D/ReadVariableOp2<
CONV2/BiasAdd/ReadVariableOpCONV2/BiasAdd/ReadVariableOp2:
CONV2/Conv2D/ReadVariableOpCONV2/Conv2D/ReadVariableOp2<
CONV3/BiasAdd/ReadVariableOpCONV3/BiasAdd/ReadVariableOp2:
CONV3/Conv2D/ReadVariableOpCONV3/Conv2D/ReadVariableOp2<
CONV4/BiasAdd/ReadVariableOpCONV4/BiasAdd/ReadVariableOp2:
CONV4/Conv2D/ReadVariableOpCONV4/Conv2D/ReadVariableOp2<
CONV5/BiasAdd/ReadVariableOpCONV5/BiasAdd/ReadVariableOp2:
CONV5/Conv2D/ReadVariableOpCONV5/Conv2D/ReadVariableOp28
FC1/BiasAdd/ReadVariableOpFC1/BiasAdd/ReadVariableOp26
FC1/MatMul/ReadVariableOpFC1/MatMul/ReadVariableOp2>
OUTPUT/BiasAdd/ReadVariableOpOUTPUT/BiasAdd/ReadVariableOp2<
OUTPUT/MatMul/ReadVariableOpOUTPUT/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
a
(__inference_DROPOUT4_layer_call_fn_27097

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_DROPOUT4_layer_call_and_return_conditional_losses_264222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
(__inference_MAXPOOL1_layer_call_fn_26217

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL1_layer_call_and_return_conditional_losses_262112
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?]
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26848

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource(
$conv5_conv2d_readvariableop_resource)
%conv5_biasadd_readvariableop_resource&
"fc1_matmul_readvariableop_resource'
#fc1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity??CONV1/BiasAdd/ReadVariableOp?CONV1/Conv2D/ReadVariableOp?CONV2/BiasAdd/ReadVariableOp?CONV2/Conv2D/ReadVariableOp?CONV3/BiasAdd/ReadVariableOp?CONV3/Conv2D/ReadVariableOp?CONV4/BiasAdd/ReadVariableOp?CONV4/Conv2D/ReadVariableOp?CONV5/BiasAdd/ReadVariableOp?CONV5/Conv2D/ReadVariableOp?FC1/BiasAdd/ReadVariableOp?FC1/MatMul/ReadVariableOp?OUTPUT/BiasAdd/ReadVariableOp?OUTPUT/MatMul/ReadVariableOp?
CONV1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
CONV1/Conv2D/ReadVariableOp?
CONV1/Conv2DConv2Dinputs#CONV1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
CONV1/Conv2D?
CONV1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
CONV1/BiasAdd/ReadVariableOp?
CONV1/BiasAddBiasAddCONV1/Conv2D:output:0$CONV1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
CONV1/BiasAddt

CONV1/ReluReluCONV1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2

CONV1/Relu?
MAXPOOL1/MaxPoolMaxPoolCONV1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
MAXPOOL1/MaxPool?
CONV2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
CONV2/Conv2D/ReadVariableOp?
CONV2/Conv2DConv2DMAXPOOL1/MaxPool:output:0#CONV2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????}} *
paddingVALID*
strides
2
CONV2/Conv2D?
CONV2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
CONV2/BiasAdd/ReadVariableOp?
CONV2/BiasAddBiasAddCONV2/Conv2D:output:0$CONV2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????}} 2
CONV2/BiasAddr

CONV2/ReluReluCONV2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}} 2

CONV2/Relu?
MAXPOOL2/MaxPoolMaxPoolCONV2/Relu:activations:0*/
_output_shapes
:?????????>> *
ksize
*
paddingVALID*
strides
2
MAXPOOL2/MaxPool?
CONV3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
CONV3/Conv2D/ReadVariableOp?
CONV3/Conv2DConv2DMAXPOOL2/MaxPool:output:0#CONV3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@*
paddingVALID*
strides
2
CONV3/Conv2D?
CONV3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
CONV3/BiasAdd/ReadVariableOp?
CONV3/BiasAddBiasAddCONV3/Conv2D:output:0$CONV3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@2
CONV3/BiasAddr

CONV3/ReluReluCONV3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<<@2

CONV3/Relu?
MAXPOOL3/MaxPoolMaxPoolCONV3/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
MAXPOOL3/MaxPool?
CONV4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
CONV4/Conv2D/ReadVariableOp?
CONV4/Conv2DConv2DMAXPOOL3/MaxPool:output:0#CONV4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
CONV4/Conv2D?
CONV4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
CONV4/BiasAdd/ReadVariableOp?
CONV4/BiasAddBiasAddCONV4/Conv2D:output:0$CONV4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
CONV4/BiasAdds

CONV4/ReluReluCONV4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2

CONV4/Relu?
MAXPOOL4/MaxPoolMaxPoolCONV4/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
MAXPOOL4/MaxPool?
CONV5/Conv2D/ReadVariableOpReadVariableOp$conv5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
CONV5/Conv2D/ReadVariableOp?
CONV5/Conv2DConv2DMAXPOOL4/MaxPool:output:0#CONV5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
CONV5/Conv2D?
CONV5/BiasAdd/ReadVariableOpReadVariableOp%conv5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
CONV5/BiasAdd/ReadVariableOp?
CONV5/BiasAddBiasAddCONV5/Conv2D:output:0$CONV5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
CONV5/BiasAdds

CONV5/ReluReluCONV5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2

CONV5/Relu?
$GLOBAL_MAXPOOL/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2&
$GLOBAL_MAXPOOL/Max/reduction_indices?
GLOBAL_MAXPOOL/MaxMaxCONV5/Relu:activations:0-GLOBAL_MAXPOOL/Max/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
GLOBAL_MAXPOOL/Maxu
DROPOUT4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
DROPOUT4/dropout/Const?
DROPOUT4/dropout/MulMulGLOBAL_MAXPOOL/Max:output:0DROPOUT4/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
DROPOUT4/dropout/Mul{
DROPOUT4/dropout/ShapeShapeGLOBAL_MAXPOOL/Max:output:0*
T0*
_output_shapes
:2
DROPOUT4/dropout/Shape?
-DROPOUT4/dropout/random_uniform/RandomUniformRandomUniformDROPOUT4/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02/
-DROPOUT4/dropout/random_uniform/RandomUniform?
DROPOUT4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2!
DROPOUT4/dropout/GreaterEqual/y?
DROPOUT4/dropout/GreaterEqualGreaterEqual6DROPOUT4/dropout/random_uniform/RandomUniform:output:0(DROPOUT4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
DROPOUT4/dropout/GreaterEqual?
DROPOUT4/dropout/CastCast!DROPOUT4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
DROPOUT4/dropout/Cast?
DROPOUT4/dropout/Mul_1MulDROPOUT4/dropout/Mul:z:0DROPOUT4/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
DROPOUT4/dropout/Mul_1?
FC1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
FC1/MatMul/ReadVariableOp?

FC1/MatMulMatMulDROPOUT4/dropout/Mul_1:z:0!FC1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

FC1/MatMul?
FC1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
FC1/BiasAdd/ReadVariableOp?
FC1/BiasAddBiasAddFC1/MatMul:product:0"FC1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
FC1/BiasAdde
FC1/ReluReluFC1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

FC1/Reluu
DROPOUT5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
DROPOUT5/dropout/Const?
DROPOUT5/dropout/MulMulFC1/Relu:activations:0DROPOUT5/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
DROPOUT5/dropout/Mulv
DROPOUT5/dropout/ShapeShapeFC1/Relu:activations:0*
T0*
_output_shapes
:2
DROPOUT5/dropout/Shape?
-DROPOUT5/dropout/random_uniform/RandomUniformRandomUniformDROPOUT5/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02/
-DROPOUT5/dropout/random_uniform/RandomUniform?
DROPOUT5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2!
DROPOUT5/dropout/GreaterEqual/y?
DROPOUT5/dropout/GreaterEqualGreaterEqual6DROPOUT5/dropout/random_uniform/RandomUniform:output:0(DROPOUT5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
DROPOUT5/dropout/GreaterEqual?
DROPOUT5/dropout/CastCast!DROPOUT5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
DROPOUT5/dropout/Cast?
DROPOUT5/dropout/Mul_1MulDROPOUT5/dropout/Mul:z:0DROPOUT5/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
DROPOUT5/dropout/Mul_1?
OUTPUT/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
OUTPUT/MatMul/ReadVariableOp?
OUTPUT/MatMulMatMulDROPOUT5/dropout/Mul_1:z:0$OUTPUT/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
OUTPUT/MatMul?
OUTPUT/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
OUTPUT/BiasAdd/ReadVariableOp?
OUTPUT/BiasAddBiasAddOUTPUT/MatMul:product:0%OUTPUT/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
OUTPUT/BiasAddv
OUTPUT/SoftmaxSoftmaxOUTPUT/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
OUTPUT/Softmax?
IdentityIdentityOUTPUT/Softmax:softmax:0^CONV1/BiasAdd/ReadVariableOp^CONV1/Conv2D/ReadVariableOp^CONV2/BiasAdd/ReadVariableOp^CONV2/Conv2D/ReadVariableOp^CONV3/BiasAdd/ReadVariableOp^CONV3/Conv2D/ReadVariableOp^CONV4/BiasAdd/ReadVariableOp^CONV4/Conv2D/ReadVariableOp^CONV5/BiasAdd/ReadVariableOp^CONV5/Conv2D/ReadVariableOp^FC1/BiasAdd/ReadVariableOp^FC1/MatMul/ReadVariableOp^OUTPUT/BiasAdd/ReadVariableOp^OUTPUT/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2<
CONV1/BiasAdd/ReadVariableOpCONV1/BiasAdd/ReadVariableOp2:
CONV1/Conv2D/ReadVariableOpCONV1/Conv2D/ReadVariableOp2<
CONV2/BiasAdd/ReadVariableOpCONV2/BiasAdd/ReadVariableOp2:
CONV2/Conv2D/ReadVariableOpCONV2/Conv2D/ReadVariableOp2<
CONV3/BiasAdd/ReadVariableOpCONV3/BiasAdd/ReadVariableOp2:
CONV3/Conv2D/ReadVariableOpCONV3/Conv2D/ReadVariableOp2<
CONV4/BiasAdd/ReadVariableOpCONV4/BiasAdd/ReadVariableOp2:
CONV4/Conv2D/ReadVariableOpCONV4/Conv2D/ReadVariableOp2<
CONV5/BiasAdd/ReadVariableOpCONV5/BiasAdd/ReadVariableOp2:
CONV5/Conv2D/ReadVariableOpCONV5/Conv2D/ReadVariableOp28
FC1/BiasAdd/ReadVariableOpFC1/BiasAdd/ReadVariableOp26
FC1/MatMul/ReadVariableOpFC1/MatMul/ReadVariableOp2>
OUTPUT/BiasAdd/ReadVariableOpOUTPUT/BiasAdd/ReadVariableOp2<
OUTPUT/MatMul/ReadVariableOpOUTPUT/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
,__inference_sequential_1_layer_call_fn_26975

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_266992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?;
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26620

inputs
conv1_26577
conv1_26579
conv2_26583
conv2_26585
conv3_26589
conv3_26591
conv4_26595
conv4_26597
conv5_26601
conv5_26603
	fc1_26608
	fc1_26610
output_26614
output_26616
identity??CONV1/StatefulPartitionedCall?CONV2/StatefulPartitionedCall?CONV3/StatefulPartitionedCall?CONV4/StatefulPartitionedCall?CONV5/StatefulPartitionedCall? DROPOUT4/StatefulPartitionedCall? DROPOUT5/StatefulPartitionedCall?FC1/StatefulPartitionedCall?OUTPUT/StatefulPartitionedCall?
CONV1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_26577conv1_26579*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV1_layer_call_and_return_conditional_losses_262812
CONV1/StatefulPartitionedCall?
MAXPOOL1/PartitionedCallPartitionedCall&CONV1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL1_layer_call_and_return_conditional_losses_262112
MAXPOOL1/PartitionedCall?
CONV2/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL1/PartitionedCall:output:0conv2_26583conv2_26585*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????}} *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV2_layer_call_and_return_conditional_losses_263092
CONV2/StatefulPartitionedCall?
MAXPOOL2/PartitionedCallPartitionedCall&CONV2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>> * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL2_layer_call_and_return_conditional_losses_262232
MAXPOOL2/PartitionedCall?
CONV3/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL2/PartitionedCall:output:0conv3_26589conv3_26591*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV3_layer_call_and_return_conditional_losses_263372
CONV3/StatefulPartitionedCall?
MAXPOOL3/PartitionedCallPartitionedCall&CONV3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL3_layer_call_and_return_conditional_losses_262352
MAXPOOL3/PartitionedCall?
CONV4/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL3/PartitionedCall:output:0conv4_26595conv4_26597*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV4_layer_call_and_return_conditional_losses_263652
CONV4/StatefulPartitionedCall?
MAXPOOL4/PartitionedCallPartitionedCall&CONV4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL4_layer_call_and_return_conditional_losses_262472
MAXPOOL4/PartitionedCall?
CONV5/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL4/PartitionedCall:output:0conv5_26601conv5_26603*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV5_layer_call_and_return_conditional_losses_263932
CONV5/StatefulPartitionedCall?
GLOBAL_MAXPOOL/PartitionedCallPartitionedCall&CONV5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_GLOBAL_MAXPOOL_layer_call_and_return_conditional_losses_262602 
GLOBAL_MAXPOOL/PartitionedCall?
 DROPOUT4/StatefulPartitionedCallStatefulPartitionedCall'GLOBAL_MAXPOOL/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_DROPOUT4_layer_call_and_return_conditional_losses_264222"
 DROPOUT4/StatefulPartitionedCall?
FC1/StatefulPartitionedCallStatefulPartitionedCall)DROPOUT4/StatefulPartitionedCall:output:0	fc1_26608	fc1_26610*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_FC1_layer_call_and_return_conditional_losses_264512
FC1/StatefulPartitionedCall?
 DROPOUT5/StatefulPartitionedCallStatefulPartitionedCall$FC1/StatefulPartitionedCall:output:0!^DROPOUT4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_DROPOUT5_layer_call_and_return_conditional_losses_264792"
 DROPOUT5/StatefulPartitionedCall?
OUTPUT/StatefulPartitionedCallStatefulPartitionedCall)DROPOUT5/StatefulPartitionedCall:output:0output_26614output_26616*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_OUTPUT_layer_call_and_return_conditional_losses_265082 
OUTPUT/StatefulPartitionedCall?
IdentityIdentity'OUTPUT/StatefulPartitionedCall:output:0^CONV1/StatefulPartitionedCall^CONV2/StatefulPartitionedCall^CONV3/StatefulPartitionedCall^CONV4/StatefulPartitionedCall^CONV5/StatefulPartitionedCall!^DROPOUT4/StatefulPartitionedCall!^DROPOUT5/StatefulPartitionedCall^FC1/StatefulPartitionedCall^OUTPUT/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2>
CONV1/StatefulPartitionedCallCONV1/StatefulPartitionedCall2>
CONV2/StatefulPartitionedCallCONV2/StatefulPartitionedCall2>
CONV3/StatefulPartitionedCallCONV3/StatefulPartitionedCall2>
CONV4/StatefulPartitionedCallCONV4/StatefulPartitionedCall2>
CONV5/StatefulPartitionedCallCONV5/StatefulPartitionedCall2D
 DROPOUT4/StatefulPartitionedCall DROPOUT4/StatefulPartitionedCall2D
 DROPOUT5/StatefulPartitionedCall DROPOUT5/StatefulPartitionedCall2:
FC1/StatefulPartitionedCallFC1/StatefulPartitionedCall2@
OUTPUT/StatefulPartitionedCallOUTPUT/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
C__inference_DROPOUT5_layer_call_and_return_conditional_losses_26479

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
#__inference_signature_wrapper_26773
conv1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_262052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_nameCONV1_input
?

?
@__inference_CONV5_layer_call_and_return_conditional_losses_26393

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?8
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26699

inputs
conv1_26656
conv1_26658
conv2_26662
conv2_26664
conv3_26668
conv3_26670
conv4_26674
conv4_26676
conv5_26680
conv5_26682
	fc1_26687
	fc1_26689
output_26693
output_26695
identity??CONV1/StatefulPartitionedCall?CONV2/StatefulPartitionedCall?CONV3/StatefulPartitionedCall?CONV4/StatefulPartitionedCall?CONV5/StatefulPartitionedCall?FC1/StatefulPartitionedCall?OUTPUT/StatefulPartitionedCall?
CONV1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_26656conv1_26658*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV1_layer_call_and_return_conditional_losses_262812
CONV1/StatefulPartitionedCall?
MAXPOOL1/PartitionedCallPartitionedCall&CONV1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL1_layer_call_and_return_conditional_losses_262112
MAXPOOL1/PartitionedCall?
CONV2/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL1/PartitionedCall:output:0conv2_26662conv2_26664*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????}} *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV2_layer_call_and_return_conditional_losses_263092
CONV2/StatefulPartitionedCall?
MAXPOOL2/PartitionedCallPartitionedCall&CONV2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????>> * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL2_layer_call_and_return_conditional_losses_262232
MAXPOOL2/PartitionedCall?
CONV3/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL2/PartitionedCall:output:0conv3_26668conv3_26670*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<<@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV3_layer_call_and_return_conditional_losses_263372
CONV3/StatefulPartitionedCall?
MAXPOOL3/PartitionedCallPartitionedCall&CONV3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL3_layer_call_and_return_conditional_losses_262352
MAXPOOL3/PartitionedCall?
CONV4/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL3/PartitionedCall:output:0conv4_26674conv4_26676*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV4_layer_call_and_return_conditional_losses_263652
CONV4/StatefulPartitionedCall?
MAXPOOL4/PartitionedCallPartitionedCall&CONV4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL4_layer_call_and_return_conditional_losses_262472
MAXPOOL4/PartitionedCall?
CONV5/StatefulPartitionedCallStatefulPartitionedCall!MAXPOOL4/PartitionedCall:output:0conv5_26680conv5_26682*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_CONV5_layer_call_and_return_conditional_losses_263932
CONV5/StatefulPartitionedCall?
GLOBAL_MAXPOOL/PartitionedCallPartitionedCall&CONV5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_GLOBAL_MAXPOOL_layer_call_and_return_conditional_losses_262602 
GLOBAL_MAXPOOL/PartitionedCall?
DROPOUT4/PartitionedCallPartitionedCall'GLOBAL_MAXPOOL/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_DROPOUT4_layer_call_and_return_conditional_losses_264272
DROPOUT4/PartitionedCall?
FC1/StatefulPartitionedCallStatefulPartitionedCall!DROPOUT4/PartitionedCall:output:0	fc1_26687	fc1_26689*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_FC1_layer_call_and_return_conditional_losses_264512
FC1/StatefulPartitionedCall?
DROPOUT5/PartitionedCallPartitionedCall$FC1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_DROPOUT5_layer_call_and_return_conditional_losses_264842
DROPOUT5/PartitionedCall?
OUTPUT/StatefulPartitionedCallStatefulPartitionedCall!DROPOUT5/PartitionedCall:output:0output_26693output_26695*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_OUTPUT_layer_call_and_return_conditional_losses_265082 
OUTPUT/StatefulPartitionedCall?
IdentityIdentity'OUTPUT/StatefulPartitionedCall:output:0^CONV1/StatefulPartitionedCall^CONV2/StatefulPartitionedCall^CONV3/StatefulPartitionedCall^CONV4/StatefulPartitionedCall^CONV5/StatefulPartitionedCall^FC1/StatefulPartitionedCall^OUTPUT/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:???????????::::::::::::::2>
CONV1/StatefulPartitionedCallCONV1/StatefulPartitionedCall2>
CONV2/StatefulPartitionedCallCONV2/StatefulPartitionedCall2>
CONV3/StatefulPartitionedCallCONV3/StatefulPartitionedCall2>
CONV4/StatefulPartitionedCallCONV4/StatefulPartitionedCall2>
CONV5/StatefulPartitionedCallCONV5/StatefulPartitionedCall2:
FC1/StatefulPartitionedCallFC1/StatefulPartitionedCall2@
OUTPUT/StatefulPartitionedCallOUTPUT/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
@__inference_CONV3_layer_call_and_return_conditional_losses_26337

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????<<@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????>> ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????>> 
 
_user_specified_nameinputs
?

?
@__inference_CONV3_layer_call_and_return_conditional_losses_27026

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<<@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????<<@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????<<@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????>> ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????>> 
 
_user_specified_nameinputs
?	
?
A__inference_OUTPUT_layer_call_and_return_conditional_losses_27160

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_GLOBAL_MAXPOOL_layer_call_fn_26266

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_GLOBAL_MAXPOOL_layer_call_and_return_conditional_losses_262602
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
C__inference_DROPOUT4_layer_call_and_return_conditional_losses_27092

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_MAXPOOL1_layer_call_and_return_conditional_losses_26211

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
x
#__inference_FC1_layer_call_fn_27122

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_FC1_layer_call_and_return_conditional_losses_264512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
C__inference_DROPOUT4_layer_call_and_return_conditional_losses_26422

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_MAXPOOL4_layer_call_and_return_conditional_losses_26247

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
D
(__inference_MAXPOOL4_layer_call_fn_26253

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_MAXPOOL4_layer_call_and_return_conditional_losses_262472
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
@__inference_CONV2_layer_call_and_return_conditional_losses_26309

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????}} *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????}} 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????}} 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????}} 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
CONV1_input>
serving_default_CONV1_input:0???????????:
OUTPUT0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:Ƴ
?p
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?k
_tf_keras_sequential?k{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "CONV1_input"}}, {"class_name": "Conv2D", "config": {"name": "CONV1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "MAXPOOL1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "CONV2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "MAXPOOL2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "CONV3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "MAXPOOL3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "CONV4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "MAXPOOL4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "CONV5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GlobalMaxPooling2D", "config": {"name": "GLOBAL_MAXPOOL", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "DROPOUT4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "FC1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "DROPOUT5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "OUTPUT", "trainable": true, "dtype": "float32", "units": 15, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "CONV1_input"}}, {"class_name": "Conv2D", "config": {"name": "CONV1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "MAXPOOL1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "CONV2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "MAXPOOL2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "CONV3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "MAXPOOL3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "CONV4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "MAXPOOL4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "CONV5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "GlobalMaxPooling2D", "config": {"name": "GLOBAL_MAXPOOL", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "DROPOUT4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "FC1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "DROPOUT5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "OUTPUT", "trainable": true, "dtype": "float32", "units": 15, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 2.685546860448085e-07, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "CONV1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "CONV1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "MAXPOOL1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "MAXPOOL1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "CONV2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "CONV2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 127, 127, 16]}}
?
%regularization_losses
&trainable_variables
'	variables
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "MAXPOOL2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "MAXPOOL2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


)kernel
*bias
+regularization_losses
,trainable_variables
-	variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "CONV3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "CONV3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62, 62, 32]}}
?
/regularization_losses
0trainable_variables
1	variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "MAXPOOL3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "MAXPOOL3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


3kernel
4bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "CONV4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "CONV4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 64]}}
?
9regularization_losses
:trainable_variables
;	variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "MAXPOOL4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "MAXPOOL4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


=kernel
>bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "CONV5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "CONV5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 128]}}
?
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GlobalMaxPooling2D", "name": "GLOBAL_MAXPOOL", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "GLOBAL_MAXPOOL", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "DROPOUT4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "DROPOUT4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

Kkernel
Lbias
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "FC1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "FC1", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "DROPOUT5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "DROPOUT5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

Ukernel
Vbias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "OUTPUT", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "OUTPUT", "trainable": true, "dtype": "float32", "units": 15, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
[iter

\beta_1

]beta_2
	^decay
_learning_ratem?m?m? m?)m?*m?3m?4m?=m?>m?Km?Lm?Um?Vm?v?v?v? v?)v?*v?3v?4v?=v?>v?Kv?Lv?Uv?Vv?"
	optimizer
?
0
1
2
 3
)4
*5
36
47
=8
>9
K10
L11
U12
V13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
 3
)4
*5
36
47
=8
>9
K10
L11
U12
V13"
trackable_list_wrapper
?
`metrics
anon_trainable_variables
trainable_variables
regularization_losses
blayer_metrics

clayers
	variables
dlayer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
&:$2CONV1/kernel
:2
CONV1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
emetrics
fnon_trainable_variables
regularization_losses
trainable_variables
glayer_metrics

hlayers
	variables
ilayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
jmetrics
knon_trainable_variables
regularization_losses
trainable_variables
llayer_metrics

mlayers
	variables
nlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$ 2CONV2/kernel
: 2
CONV2/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
ometrics
pnon_trainable_variables
!regularization_losses
"trainable_variables
qlayer_metrics

rlayers
#	variables
slayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
tmetrics
unon_trainable_variables
%regularization_losses
&trainable_variables
vlayer_metrics

wlayers
'	variables
xlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$ @2CONV3/kernel
:@2
CONV3/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
ymetrics
znon_trainable_variables
+regularization_losses
,trainable_variables
{layer_metrics

|layers
-	variables
}layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
~metrics
non_trainable_variables
/regularization_losses
0trainable_variables
?layer_metrics
?layers
1	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%@?2CONV4/kernel
:?2
CONV4/bias
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
?metrics
?non_trainable_variables
5regularization_losses
6trainable_variables
?layer_metrics
?layers
7	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
9regularization_losses
:trainable_variables
?layer_metrics
?layers
;	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&??2CONV5/kernel
:?2
CONV5/bias
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
?metrics
?non_trainable_variables
?regularization_losses
@trainable_variables
?layer_metrics
?layers
A	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
Cregularization_losses
Dtrainable_variables
?layer_metrics
?layers
E	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
Gregularization_losses
Htrainable_variables
?layer_metrics
?layers
I	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
??2
FC1/kernel
:?2FC1/bias
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
?
?metrics
?non_trainable_variables
Mregularization_losses
Ntrainable_variables
?layer_metrics
?layers
O	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?non_trainable_variables
Qregularization_losses
Rtrainable_variables
?layer_metrics
?layers
S	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	?2OUTPUT/kernel
:2OUTPUT/bias
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
?
?metrics
?non_trainable_variables
Wregularization_losses
Xtrainable_variables
?layer_metrics
?layers
Y	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2iter
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
&:$2CONV1/kernel/m
:2CONV1/bias/m
&:$ 2CONV2/kernel/m
: 2CONV2/bias/m
&:$ @2CONV3/kernel/m
:@2CONV3/bias/m
':%@?2CONV4/kernel/m
:?2CONV4/bias/m
(:&??2CONV5/kernel/m
:?2CONV5/bias/m
:
??2FC1/kernel/m
:?2
FC1/bias/m
 :	?2OUTPUT/kernel/m
:2OUTPUT/bias/m
&:$2CONV1/kernel/v
:2CONV1/bias/v
&:$ 2CONV2/kernel/v
: 2CONV2/bias/v
&:$ @2CONV3/kernel/v
:@2CONV3/bias/v
':%@?2CONV4/kernel/v
:?2CONV4/bias/v
(:&??2CONV5/kernel/v
:?2CONV5/bias/v
:
??2FC1/kernel/v
:?2
FC1/bias/v
 :	?2OUTPUT/kernel/v
:2OUTPUT/bias/v
?2?
,__inference_sequential_1_layer_call_fn_26651
,__inference_sequential_1_layer_call_fn_26730
,__inference_sequential_1_layer_call_fn_26942
,__inference_sequential_1_layer_call_fn_26975?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_26205?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *4?1
/?,
CONV1_input???????????
?2?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26848
G__inference_sequential_1_layer_call_and_return_conditional_losses_26571
G__inference_sequential_1_layer_call_and_return_conditional_losses_26525
G__inference_sequential_1_layer_call_and_return_conditional_losses_26909?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_CONV1_layer_call_fn_26995?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_CONV1_layer_call_and_return_conditional_losses_26986?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_MAXPOOL1_layer_call_fn_26217?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_MAXPOOL1_layer_call_and_return_conditional_losses_26211?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
%__inference_CONV2_layer_call_fn_27015?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_CONV2_layer_call_and_return_conditional_losses_27006?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_MAXPOOL2_layer_call_fn_26229?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_MAXPOOL2_layer_call_and_return_conditional_losses_26223?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
%__inference_CONV3_layer_call_fn_27035?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_CONV3_layer_call_and_return_conditional_losses_27026?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_MAXPOOL3_layer_call_fn_26241?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_MAXPOOL3_layer_call_and_return_conditional_losses_26235?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
%__inference_CONV4_layer_call_fn_27055?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_CONV4_layer_call_and_return_conditional_losses_27046?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_MAXPOOL4_layer_call_fn_26253?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_MAXPOOL4_layer_call_and_return_conditional_losses_26247?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
%__inference_CONV5_layer_call_fn_27075?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_CONV5_layer_call_and_return_conditional_losses_27066?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_GLOBAL_MAXPOOL_layer_call_fn_26266?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
I__inference_GLOBAL_MAXPOOL_layer_call_and_return_conditional_losses_26260?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_DROPOUT4_layer_call_fn_27102
(__inference_DROPOUT4_layer_call_fn_27097?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_DROPOUT4_layer_call_and_return_conditional_losses_27092
C__inference_DROPOUT4_layer_call_and_return_conditional_losses_27087?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference_FC1_layer_call_fn_27122?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_FC1_layer_call_and_return_conditional_losses_27113?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_DROPOUT5_layer_call_fn_27149
(__inference_DROPOUT5_layer_call_fn_27144?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_DROPOUT5_layer_call_and_return_conditional_losses_27134
C__inference_DROPOUT5_layer_call_and_return_conditional_losses_27139?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_OUTPUT_layer_call_fn_27169?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_OUTPUT_layer_call_and_return_conditional_losses_27160?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_26773CONV1_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
@__inference_CONV1_layer_call_and_return_conditional_losses_26986p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
%__inference_CONV1_layer_call_fn_26995c9?6
/?,
*?'
inputs???????????
? ""?????????????
@__inference_CONV2_layer_call_and_return_conditional_losses_27006l 7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????}} 
? ?
%__inference_CONV2_layer_call_fn_27015_ 7?4
-?*
(?%
inputs?????????
? " ??????????}} ?
@__inference_CONV3_layer_call_and_return_conditional_losses_27026l)*7?4
-?*
(?%
inputs?????????>> 
? "-?*
#? 
0?????????<<@
? ?
%__inference_CONV3_layer_call_fn_27035_)*7?4
-?*
(?%
inputs?????????>> 
? " ??????????<<@?
@__inference_CONV4_layer_call_and_return_conditional_losses_27046m347?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
%__inference_CONV4_layer_call_fn_27055`347?4
-?*
(?%
inputs?????????@
? "!????????????
@__inference_CONV5_layer_call_and_return_conditional_losses_27066n=>8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
%__inference_CONV5_layer_call_fn_27075a=>8?5
.?+
)?&
inputs??????????
? "!????????????
C__inference_DROPOUT4_layer_call_and_return_conditional_losses_27087^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
C__inference_DROPOUT4_layer_call_and_return_conditional_losses_27092^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? }
(__inference_DROPOUT4_layer_call_fn_27097Q4?1
*?'
!?
inputs??????????
p
? "???????????}
(__inference_DROPOUT4_layer_call_fn_27102Q4?1
*?'
!?
inputs??????????
p 
? "????????????
C__inference_DROPOUT5_layer_call_and_return_conditional_losses_27134^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
C__inference_DROPOUT5_layer_call_and_return_conditional_losses_27139^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? }
(__inference_DROPOUT5_layer_call_fn_27144Q4?1
*?'
!?
inputs??????????
p
? "???????????}
(__inference_DROPOUT5_layer_call_fn_27149Q4?1
*?'
!?
inputs??????????
p 
? "????????????
>__inference_FC1_layer_call_and_return_conditional_losses_27113^KL0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? x
#__inference_FC1_layer_call_fn_27122QKL0?-
&?#
!?
inputs??????????
? "????????????
I__inference_GLOBAL_MAXPOOL_layer_call_and_return_conditional_losses_26260?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
.__inference_GLOBAL_MAXPOOL_layer_call_fn_26266wR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????????????
C__inference_MAXPOOL1_layer_call_and_return_conditional_losses_26211?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
(__inference_MAXPOOL1_layer_call_fn_26217?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_MAXPOOL2_layer_call_and_return_conditional_losses_26223?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
(__inference_MAXPOOL2_layer_call_fn_26229?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_MAXPOOL3_layer_call_and_return_conditional_losses_26235?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
(__inference_MAXPOOL3_layer_call_fn_26241?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_MAXPOOL4_layer_call_and_return_conditional_losses_26247?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
(__inference_MAXPOOL4_layer_call_fn_26253?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
A__inference_OUTPUT_layer_call_and_return_conditional_losses_27160]UV0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? z
&__inference_OUTPUT_layer_call_fn_27169PUV0?-
&?#
!?
inputs??????????
? "???????????
 __inference__wrapped_model_26205? )*34=>KLUV>?;
4?1
/?,
CONV1_input???????????
? "/?,
*
OUTPUT ?
OUTPUT??????????
G__inference_sequential_1_layer_call_and_return_conditional_losses_26525 )*34=>KLUVF?C
<?9
/?,
CONV1_input???????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26571 )*34=>KLUVF?C
<?9
/?,
CONV1_input???????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26848z )*34=>KLUVA?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_26909z )*34=>KLUVA?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
,__inference_sequential_1_layer_call_fn_26651r )*34=>KLUVF?C
<?9
/?,
CONV1_input???????????
p

 
? "???????????
,__inference_sequential_1_layer_call_fn_26730r )*34=>KLUVF?C
<?9
/?,
CONV1_input???????????
p 

 
? "???????????
,__inference_sequential_1_layer_call_fn_26942m )*34=>KLUVA?>
7?4
*?'
inputs???????????
p

 
? "???????????
,__inference_sequential_1_layer_call_fn_26975m )*34=>KLUVA?>
7?4
*?'
inputs???????????
p 

 
? "???????????
#__inference_signature_wrapper_26773? )*34=>KLUVM?J
? 
C?@
>
CONV1_input/?,
CONV1_input???????????"/?,
*
OUTPUT ?
OUTPUT?????????