//
//  ChooseTestCaseVC.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2018-01-17.
//  Copyright © 2018 AHN. All rights reserved.
//

#import "Globals.h"
#import "ChooseTestCaseVC.h"

#define ROWHEIGHT 100

// Table View Cell
//=============================================
@implementation ChooseTestCaseCell
//-------------------------------------------------------------------------------------------------------
- (instancetype)initWithStyle:(UITableViewCellStyle)style reuseIdentifier:(NSString *)reuseIdentifier
{
    self = [super initWithStyle:style reuseIdentifier:reuseIdentifier];
    if (self) {
        self.selectionStyle = UITableViewCellSelectionStyleNone;
        
        self.backgroundColor = [UIColor clearColor];
        
        self.textLabel.font = [UIFont boldSystemFontOfSize:16.0];
        self.textLabel.textColor = [UIColor whiteColor];
    }
    return self;
}

//------------------------
- (void)layoutSubviews
{
    [super layoutSubviews];
    CGRect frame = self.frame;
    frame.size.height = ROWHEIGHT - 10;
    self.frame = frame;
}

//----------------------------------------------------------------
- (void)setHighlighted:(BOOL)highlighted animated:(BOOL)animated
{
    self.textLabel.alpha = highlighted ? 0.5 : 1.0;
}
@end // ChooseTestCaseCell


// Table View Controller
//=====================================================
@interface ChooseTestCaseVC ()
@property (strong, nonatomic) NSArray *titlesArray;
@end

@implementation ChooseTestCaseVC

//----------
- (id)init
{
    self = [super initWithStyle:UITableViewStylePlain];
    if (self) {
//        self.titlesArray = [@[@"Item 1"
//                             ,@"Item 2"
//                             ] mutableCopy];
        
        self.view.backgroundColor = [UIColor clearColor];
        
        [self.tableView registerClass:[ChooseTestCaseCell class] forCellReuseIdentifier:@"cell"];
        self.tableView.separatorStyle = UITableViewCellSeparatorStyleNone;
        //self.tableView.contentInset = UIEdgeInsetsMake(44.0, 0.0, 44.0, 0.0);
        self.tableView.showsVerticalScrollIndicator = NO;
        self.tableView.backgroundColor = [UIColor clearColor];
        self.tableView.rowHeight = 60;
    }
    return self;
}
//-------------------------------------------
- (void) viewWillAppear:(BOOL) animated
{
    [super viewWillAppear: animated];
    NSArray *files = glob_files( @"", @TESTCASE_PREFIX, @".jpg");
    self.titlesArray = files;
}
//-------------------------------
- (BOOL)prefersStatusBarHidden
{
    return YES;
}
//--------------------------------------------
- (UIStatusBarStyle)preferredStatusBarStyle
{
    return UIStatusBarStyleDefault;
}
//-----------------------------------------------------------
- (UIStatusBarAnimation)preferredStatusBarUpdateAnimation
{
    return UIStatusBarAnimationFade;
}
#pragma mark - UITableViewDataSource
//-----------------------------------------------------------------
- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView
{
    return 1;
}
//------------------------------------------------------------------------------------------
- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section
{
    return self.titlesArray.count;
}
//------------------------------------------------------------------------------------------------------
- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath
{
    ChooseTestCaseCell *cell = [tableView dequeueReusableCellWithIdentifier:@"cell"];
    
    cell.textLabel.text = self.titlesArray[indexPath.row];
    cell.imageView.image = [UIImage imageNamed:@"board01.jpg"];
    //cell.separatorView.hidden = (indexPath.row <= 3 || indexPath.row == self.titlesArray.count-1);
    //cell.userInteractionEnabled = (indexPath.row != 1 && indexPath.row != 3);
    
    return cell;
}
#pragma mark - UITableViewDelegate
//-----------------------------------------------------------------------------------------------
- (CGFloat)tableView:(UITableView *)tableView heightForRowAtIndexPath:(NSIndexPath *)indexPath
{
    return ROWHEIGHT;
}
//--------------------------------------------------------------------------------------------
- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath
{
    popup( @"ChooseTestCase click", @"");
//    NSString *menuItem = _titlesArray[indexPath.row];
//    if ([menuItem hasPrefix:@"Save as Test Case"]) {
//        [g_app.mainVC mnuSaveAsTestCase];
//    }
//    else if ([menuItem hasPrefix:@"Set Current Test Case"]) {
//        [g_app.mainVC mnuSetCurrentTestCase];
//    }
}

@end

