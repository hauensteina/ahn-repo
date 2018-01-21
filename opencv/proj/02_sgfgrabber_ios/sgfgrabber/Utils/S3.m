//
//  S3.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2018-01-21.
//  Copyright © 2018 AHN. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <AWSCore/AWSCore.h>
#import <AWSCognito/AWSCognito.h>
#import <AWSS3/AWSS3.h>
#import "Common.h"

#define BUCKET_NAME @"kifu-cam"
static AWSCognitoCredentialsProvider *s_credentialsProvider = nil;

// Authenticate with AWS for access to kifu-cam bucket
//-------------------------------------------------------
void S3_login()
{
    if (s_credentialsProvider) return;
    s_credentialsProvider = [[AWSCognitoCredentialsProvider alloc]
                             initWithRegionType:AWSRegionUSWest2
                             identityPoolId:@"us-west-2:86844471-fec8-4356-a48d-2cb7c620b97a"];
    
    AWSServiceConfiguration *configuration = [[AWSServiceConfiguration alloc] initWithRegion:AWSRegionUSWest2 credentialsProvider:s_credentialsProvider];
    
    [AWSServiceManager defaultServiceManager].defaultServiceConfiguration = configuration;
} // S3_login()

// Upload a file to kifu-cam bucket
//-------------------------------------
void S3_upload_file( NSString *fname)
{
    NSString *fullfname = getFullPath( fname);
    NSURL *uploadingFileURL = [NSURL fileURLWithPath: fullfname];
    
    AWSS3TransferManagerUploadRequest *uploadRequest = [AWSS3TransferManagerUploadRequest new];
    
    uploadRequest.bucket = BUCKET_NAME;
    uploadRequest.key = fname;
    uploadRequest.body = uploadingFileURL;
    
    AWSS3TransferManager *transferManager = [AWSS3TransferManager defaultS3TransferManager];
    [[transferManager upload:uploadRequest]
     continueWithExecutor:[AWSExecutor mainThreadExecutor]
     withBlock:^id(AWSTask *task) {
         if (task.error) {
             if ([task.error.domain isEqualToString:AWSS3TransferManagerErrorDomain]) {
                 switch (task.error.code) {
                     case AWSS3TransferManagerErrorCancelled:
                     case AWSS3TransferManagerErrorPaused:
                         break;
                         
                     default:
                         NSLog(@"Error: %@", task.error);
                         popup( nsprintf( @"S3 upload failed for %@. Error:%@", uploadRequest.key, task.error), @"Error");
                         break;
                 }
             } else {
                 // Unknown error.
                 popup( nsprintf( @"S3 upload failed for %@. Error:%@", uploadRequest.key, task.error), @"Error");
                 NSLog(@"Error: %@", task.error);
             }
         }
         
         if (task.result) {
             // Success
             //AWSS3TransferManagerUploadOutput *uploadOutput = task.result;
         }
         return nil;
     }];
} // upload_file()

