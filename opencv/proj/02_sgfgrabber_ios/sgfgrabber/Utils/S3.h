//
//  S3.h
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2018-01-21.
//  Copyright © 2018 AHN. All rights reserved.
//

// Helper functions for S3 uploads and downloads

#ifndef S3_h
#define S3_h

// Authenticate with AWS for access to kifu-cam bucket
void S3_login(void);
void S3_upload_file( NSString *fname);
#endif /* S3_h */

