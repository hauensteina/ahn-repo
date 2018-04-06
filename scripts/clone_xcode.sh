#!/bin/sh

# Script to clone an xcode app with a new name,
# as a starting point for an OSX project.

# AHN, Mar 2013

function usage()
{
    echo " "
    echo "usage: clone_xcode.sh <oldAppName> <newAppName>"
    echo "Makes a copy of a folder containing an xcode project"
    echo "and replaces all strings to change the name" 
    echo " "
    exit 1
}


if [ $# -ne 2 ] ; then
    usage
fi

oldAppName=$1
newAppName=$2

if [ ! -d $oldAppName ] ; then
    echo "$oldAppName does not exist or is not a folder"
    exit 1
fi

if [ -e $newAppName ] ; then
    echo "$newAppName already exists. Please remove first."
    exit 1
fi

echo "copying $oldAppName to $newAppName"
cp -R "$oldAppName" "$newAppName"

cd "$newAppName"

echo "Removing DerivedData ..."
rm -rf `find . -name DerivedData`
echo "Removing build ..."
rm -rf `find . -name build`
echo "Removing xcuserdata ..."
rm -rf `find . -name xcuserdata`

echo "renaming xcodeproj and project folder"
mv "$oldAppName" "$newAppName"
mv "$oldAppName.xcodeproj" "$newAppName.xcodeproj"

echo "renaming other files"
appNamedFiles=`find . -name "*$oldAppName*"`
for f in $appNamedFiles; do
    newf=`echo "$f" | sed s/$oldAppName/$newAppName/g`
    if [ -e "$newf" ] ; then
	echo "Error: $newf already exists in $newAppName. Exiting."
        exit 1
    fi
    echo "moving $f to $newf" 
    mv "$f" "$newf"
done

echo "replacing strings inside files"
find . -type f -exec sed -i "" "s/$oldAppName/$newAppName/g" "{}" \; 2>/dev/null

cd ..
echo "Done cloning $oldAppName to $newAppName"
echo "Good luck!"
 



