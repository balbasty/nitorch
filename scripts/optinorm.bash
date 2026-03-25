#! /bin/bash

# usage:
#    <script> INPUTS... [-s STEPS...] [-o OUTPUT] [-m MEMBRANE]
#
# arguments:
#    INPUTS    paths to input nifti files
#    STEPS     ("aff" "affopt" "meanspace" "svf" "svfopt" "norm")
#    OUTPUT    path to output folder
#    MEMBRANE  membrane energy regularizaton (20)
#
# author:
#    Yael Balbastre - March 3rd, 2025

# Default arguments
INPUTS=()
STEPS=("aff" "affopt" "meanspace" "svf" "svfopt" "norm")
OUT="norm"
MEMBRANE=20

# Parse arguments
DOSTEP=""
while [[ ! -z $1 ]]; do
case $1 in
"-s") STEPS=()      && DOSTEP=1;;
"-o") OUT=$2        && DOSTEP="" && shift;;
"-m") MEMBRANE=$2   && DOSTEP="" && shift;;
   *) [[ -z $DOSTEP ]] && INPUTS+=($1) || STEPS+=($1);;
esac
shift
done

N=${#INPUTS[@]}

if (( $N == 0 )); then
echo "No input files provided"
echo "usage:"
echo "   <script> INPUTS... [-s STEPS...] [-o OUTPUT] [-m MEMBRANE]"
echo ""
echo "arguments:"
echo "   INPUTS    paths to input nifti files"
echo "   STEPS     ("aff" "affopt" "meanspace" "svf" "svfopt" "norm")"
echo "   OUTPUT    path to output folder"
echo "   MEMBRANE  membrane energy regularizaton (20)"
exit 1
fi

echo $0 ${INPUTS[@]} "-o" $OUT "-s" ${STEPS[@]}

# Output directory structure
AFF=${OUT}/aff
SVF=${OUT}/svf
OPT=${OUT}/opt
OPTAFF=${OPT}/aff
OPTSVF=${OPT}/svf
OPTSVFW=${OPTSVF}/weighted
OPTSVFU=${OPTSVF}/unweighted

[[ -d $OUT ]]     || mkdir $OUT
[[ -d $AFF ]]     || mkdir $AFF
[[ -d $SVF ]]     || mkdir $SVF
[[ -d $OPT ]]     || mkdir $OPT
[[ -d $OPTAFF ]]  || mkdir $OPTAFF
[[ -d $OPTSVF ]]  || mkdir $OPTSVF
[[ -d $OPTSVFU ]] || mkdir $OPTSVFU
[[ -d $OPTSVFW ]] || mkdir $OPTSVFW

array_contains () {
    local array="$1[@]"
    local seeking=$2
    local in=1
    for element in "${!array}"; do
        if [[ $element == "$seeking" ]]; then
            in=0
            break
        fi
    done
    return $in
}

# ----------------------------------------
# Affine pairwises
if array_contains STEPS "aff"; then

for i in ${!INPUTS[@]}; do
for j in ${!INPUTS[@]}; do
(( $i == $j )) && continue
nitorch register -v \
	-o ${AFF}/$((i+1))$((j+1)) \
	@loss lcc \
	@@mov ${INPUTS[$i]} \
	@@fix ${INPUTS[$j]} \
	@affine rigid -g \
	@pyramid --levels :4
done
done

fi

# ----------------------------------------
# Optimal mean-to-session affine
if array_contains STEPS "affopt"; then

ARGS=()
for i in ${!INPUTS[@]}; do
for j in ${!INPUTS[@]}; do
[[ $i == $j ]] && continue
ARGS+=(-i $((i+1)) $((j+1)) ${AFF}/$((i+1))$((j+1))/rigid.lta)
done
done

nitorch affopt -r ${ARGS[@]} \
	-o "${OPTAFF}/rigid_optim_{label}.lta"


# Inverse optimal affines (for freeview)
for i in ${!INPUTS[@]}; do
nitorch compose \
	-il ${OPTAFF}/rigid_optim_$((i+1)).lta \
	-o ${OPTAFF}/rigid_optim_$((i+1))_inv.lta
done

# Apply optimal affines (without reslicing)
for i in ${!INPUTS[@]}; do
nitorch reslice \
	${INPUTS[$i]} \
	-l ${OPTAFF}/rigid_optim_$((i+1)).lta \
	-o ${OPT}/{base}.aligned{ext}
done

fi

# ----------------------------------------
# Compute mean space for nonlinear registration
if array_contains STEPS "meanspace"; then

nitorch meanspace ${OPT}/*.aligned.nii.gz -o ${OPT}/meanspace.nii.gz

# Reslice to meanspace (for viz)
for i in ${!INPUTS[@]}; do
nitorch reslice \
	${INPUTS[$i]} \
	-l ${OPTAFF}/rigid_optim_$((i+1)).lta \
	-t ${OPT}/meanspace.nii.gz \
	-o ${OPT}/{base}.aligned.resliced{ext} \
	-i 3 --clip
done

fi

# ----------------------------------------
# Perform pairwise nonlinear registration
if array_contains STEPS "svf"; then

for i in ${!INPUTS[@]}; do
for j in ${!INPUTS[@]}; do
[[ $i == $j ]] && continue
nitorch register -v 2 \
        -o ${SVF}/$((i+1))$((j+1)) \
        @loss lcc  \
        @@mov ${INPUTS[$i]} -a ${OPTAFF}/rigid_optim_$((i+1)).lta \
        @@fix ${INPUTS[$j]} -a ${OPTAFF}/rigid_optim_$((j+1)).lta \
        @nonlin -a 0 -m $MEMBRANE -b 0 -l 0 \
	-i ${OPT}/meanspace.nii.gz \
	-h "{dir}{sep}hessian.nii.gz" \
        @pyramid --levels :6
done
done

fi

# ----------------------------------------
if array_contains STEPS "svfopt"; then
# Optimal mean-to-session affine

# Hessian-weighted
ARGS=()
for i in ${!INPUTS[@]}; do
for j in ${!INPUTS[@]}; do
[[ $i == $j ]] && continue
ARGS+=(-i $((i+1)) $((j+1)) ${SVF}/$((i+1))$((j+1))/svf.nii.gz)
ARGS+=(-w $((i+1)) $((j+1)) ${SVF}/$((i+1))$((j+1))/hessian.nii.gz)
done
done

nitorch vopt ${ARGS[@]} \
-a 0 -m $MEMBRANE -b 0 -l 0 0 \
-o "${OPTSVF}/weighted/svf_optim_weighted_{label}.nii.gz"


# Unweighted
ARGS=()
for i in ${!INPUTS[@]}; do
for j in ${!INPUTS[@]}; do
[[ $i == $j ]] && continue
ARGS+=(-i $((i+1)) $((j+1)) ${SVF}/$((i+1))$((j+1))/svf.nii.gz)
done
done

nitorch vopt ${ARGS[@]} \
-a 0 -m $MEMBRANE -b 0 -l 0 0 \
-o "${OPTSVF}/unweighted/svf_optim_unweighted_{label}.nii.gz"

fi


# ----------------------------------------
if array_contains STEPS "norm"; then

for i in ${!INPUTS[@]}; do
nitorch reslice \
	${INPUTS[$i]} \
	-l "${OPTAFF}/rigid_optim_$((i+1)).lta" \
	-v "${OPTSVF}/weighted/svf_optim_weighted_$((i+1)).nii.gz" \
	-t ${OPT}/meanspace.nii.gz \
	-o "${OPT}/{base}.norm{ext}" \
	-i 3 --clip
done

fi
