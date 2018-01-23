//
//  ImagesVC.h
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2018-01-17.
//  Copyright © 2018 AHN. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface ImagesVC : UITableViewController
- (void)refresh;
@property NSString *selectedImageName;
@end

@interface ImagesCell : UITableViewCell
@end

