//
//  EditTestCaseVC.m
//  sgfgrabber
//
//  Created by Andreas Hauenstein on 2018-01-17.
//  Copyright © 2018 AHN. All rights reserved.
//

#import "Globals.h"
#import "EditTestCaseVC.h"
#import "CppInterface.h"

#define ROWHEIGHT 140

// Table View Cell
//=============================================
@implementation EditTestCaseCell
//-------------------------------------------------------------------------------------------------------
- (instancetype)initWithStyle:(UITableViewCellStyle)style reuseIdentifier:(NSString *)reuseIdentifier
{
    self = [super initWithStyle:style reuseIdentifier:reuseIdentifier];
    if (self) {
        self.selectionStyle = UITableViewCellSelectionStyleNone;
        
        self.backgroundColor = [UIColor clearColor];
        
        self.textLabel.font = [UIFont boldSystemFontOfSize:16.0];
        self.textLabel.textColor = [UIColor whiteColor];
        self.textLabel.backgroundColor = [UIColor clearColor];
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
@end // EditTestCaseCell


// Table View Controller
//=====================================================
@interface EditTestCaseVC ()
@property (strong, nonatomic) NSArray *titlesArray;
@property long selected_row;
@property long highlighted_row;
@end

@implementation EditTestCaseVC

//----------
- (id)init
{
    self = [super initWithStyle:UITableViewStylePlain];
    if (self) {
        self.view.backgroundColor = [UIColor clearColor];
        
        [self.tableView registerClass:[EditTestCaseCell class] forCellReuseIdentifier:@"cell"];
        self.tableView.separatorStyle = UITableViewCellSeparatorStyleNone;
        self.tableView.showsVerticalScrollIndicator = NO;
        self.tableView.backgroundColor = [UIColor clearColor];
        //self.tableView.rowHeight = 150;
    }
    return self;
}

//---------------------------------------
- (void)refresh
{
    [self loadTitlesArray];
    [self.tableView reloadData];
    [self.tableView reloadSections:[NSIndexSet indexSetWithIndex:0] withRowAnimation:UITableViewRowAnimationNone];
}

// Find testcase files and remember their names for display. One per row.
//-------------------------
- (void) loadTitlesArray
{
    self.titlesArray = glob_files( @"", @TESTCASE_PREFIX, @".jpg");
}

//-------------------------------------------
- (void) viewWillAppear:(BOOL) animated
{
    [super viewWillAppear: animated];
    [self loadTitlesArray];
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
- (EditTestCaseCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath
{
    EditTestCaseCell *cell = [tableView dequeueReusableCellWithIdentifier:@"cell"];
    [[cell subviews] makeObjectsPerformSelector:@selector(removeFromSuperview)];
    NSString *fname = self.titlesArray[indexPath.row];
    // Photo
    UIImageView *imgView1 = [[UIImageView alloc] initWithFrame:CGRectMake(40,20,70,70)];
    NSString *fullfname = getFullPath( fname);
    UIImage *img = [UIImage imageWithContentsOfFile:fullfname];
    imgView1.image = img;
    [cell addSubview: imgView1];
    // Diagram
    UIImageView *imgView2 = [[UIImageView alloc] initWithFrame:CGRectMake(140,20,70,70)];
    fullfname = changeExtension( fullfname, @".sgf");
    NSString *sgf = [NSString stringWithContentsOfFile:fullfname encoding:NSUTF8StringEncoding error:NULL];
    UIImage *sgfImg = [CppInterface sgf2img:sgf];
    imgView2.image = sgfImg;
    [cell addSubview: imgView2];
    // Name
    UILabel *lb = [[UILabel alloc] initWithFrame:CGRectMake(40,70,250,70)];
    lb.text = fname;
    [cell addSubview:lb];
    //cell.backgroundColor = self.view.tintColor;
    cell.backgroundColor = [UIColor clearColor];
    if (indexPath.row == _highlighted_row) {
        cell.backgroundColor = self.view.tintColor;
    }
    return cell;
}
#pragma mark - UITableViewDelegate
//-----------------------------------------------------------------------------------------------
- (CGFloat)tableView:(UITableView *)tableView heightForRowAtIndexPath:(NSIndexPath *)indexPath
{
    return ROWHEIGHT;
}

// Click on Test Case
//--------------------------------------------------------------------------------------------
- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath
{
    _selected_row = indexPath.row;
    NSArray *choices = @[@"Select", @"Pair with current position", @"Delete", @"Cancel"];
    choicePopup( choices, @"Action",
                ^(UIAlertAction *action) {
                    [self handleEditAction:action.title];
                });
}

// Handle test case edit action
//---------------------------------------------
- (void)handleEditAction:(NSString *)action
{
    if ([action hasPrefix:@"Select"]) {
        [self handleMakeCurrentAction];
    }
    else if ([action hasPrefix:@"Pair with current position"]) {
        
    }
    else if ([action hasPrefix:@"Delete"]) {
        NSString *fname = _titlesArray[_selected_row];
        fname = getFullPath( fname);
        choicePopup(@[@"Delete",@"Cancel"], @"Really?",
                    ^(UIAlertAction *action) {
                        [self handleDeleteAction:action.title];
                    });
    }
    else {}
} // handleEditAction()

// Delete current test case
//---------------------------------------------
- (void)handleDeleteAction:(NSString *)action
{
    if (![action hasPrefix:@"Delete"]) return;
    // Delete jpg file
    NSString *fname = _titlesArray[_selected_row];
    fname = getFullPath( fname);
    unlink( [fname UTF8String]);
    // Delete sgf file
    fname = changeExtension( fname, @".sgf");
    unlink( [fname UTF8String]);
    [self refresh];
} // handleDeleteAction()

// Set current test case
//---------------------------------
- (void)handleMakeCurrentAction
{
    NSString *fname = _titlesArray[_selected_row];
    _selectedTestCase = fname;
    _highlighted_row = _selected_row;
    //popup( nsprintf( @"%@ selected", fname), @"");
    [self refresh];
} // handleDeleteAction()


























@end // EditTestCaseVC

