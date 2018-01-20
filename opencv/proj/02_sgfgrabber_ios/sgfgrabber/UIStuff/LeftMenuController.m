//
//  LeftViewController.m
//  LGSideMenuControllerDemo
//

#import "Globals.h"
#import "CppInterface.h"
#import "LeftMenuController.h"
#import "LeftMenuCell.h"
#import "TopViewController.h"
#import "UIViewController+LGSideMenuController.h"

enum {ITEM_NOT_SELECTED=0, ITEM_SELECTED=1};
enum {VIDEO_MODE=0, PHOTO_MODE=1, DEBUG_MODE=2};

@interface LeftMenuController ()
@property (strong, nonatomic) NSMutableArray *titlesArray;
@property UIFont *normalFont;
@property UIFont *selectedFont;
@property int mode;
@end

@implementation LeftMenuController
// Initialize left menu
//-----------------------
- (id)init
{
    self = [super initWithStyle:UITableViewStylePlain];
    if (self) {
        _selectedFont = [UIFont fontWithName:@"Verdana-Bold" size:16 ];
        _normalFont = [UIFont fontWithName:@"Verdana" size:16 ];
        NSArray *d = @[
                       @{ @"txt": @"Video Mode", @"state": @(ITEM_SELECTED) },
                       @{ @"txt": @"Photo Mode", @"state": @(ITEM_NOT_SELECTED) },
                       @{ @"txt": @"Debug Mode", @"state": @(ITEM_NOT_SELECTED) },
                       @{ @"txt": @"", @"state": @(ITEM_NOT_SELECTED) },
                       @{ @"txt": @"Add Test Case", @"state": @(ITEM_NOT_SELECTED) },
                       @{ @"txt": @"Edit Test Cases", @"state": @(ITEM_NOT_SELECTED) },
                       @{ @"txt": @"Run Test Cases", @"state": @(ITEM_NOT_SELECTED) },
                       ];
        _titlesArray = [NSMutableArray new];
        for (NSDictionary *x in d) { [_titlesArray addObject:[x mutableCopy]]; }
        
        self.view.backgroundColor = [UIColor clearColor];

        [self.tableView registerClass:[LeftMenuCell class] forCellReuseIdentifier:@"cell"];
        self.tableView.separatorStyle =  UITableViewCellSeparatorStyleNone;
        self.tableView.contentInset = UIEdgeInsetsMake(44.0, 0.0, 44.0, 0.0);
        self.tableView.showsVerticalScrollIndicator = NO;
        self.tableView.backgroundColor = [UIColor clearColor];
    }
    return self;
} // init()

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
    LeftMenuCell *cell = [tableView dequeueReusableCellWithIdentifier:@"cell"];
    cell.textLabel.text = self.titlesArray[indexPath.row][@"txt"];
    if ([_titlesArray[indexPath.row][@"state"] intValue] == ITEM_SELECTED) {
        [cell.textLabel setFont:_selectedFont];
    }
    else {
        [cell.textLabel setFont:_normalFont];
    }
    return cell;
} // cellForRowAtIndexPath()

#pragma mark - UITableViewDelegate
//-----------------------------------------------------------------------------------------------
- (CGFloat)tableView:(UITableView *)tableView heightForRowAtIndexPath:(NSIndexPath *)indexPath
{
    return 50;
}

//-------------------
- (void)unselectAll
{
    for (id x in _titlesArray) { x[@"state"] = @(ITEM_NOT_SELECTED); }
}

//-------------------------------------------------
- (void)setState:(int)state forRow:(NSInteger)row
{
    _titlesArray[row][@"state"] = @(state);
}

- (bool) videoMode { return _mode == VIDEO_MODE; }
- (bool) photoMode { return _mode == PHOTO_MODE; }
- (bool) debugMode { return _mode == DEBUG_MODE; }

// Handle left menu choice
//--------------------------------------------------------------------------------------------
- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath
{
    TopViewController *topViewController = (TopViewController *)self.sideMenuController;
    NSString *menuItem = _titlesArray[indexPath.row][@"txt"];

    if ([menuItem hasPrefix:@"Edit Test Cases"]) {
        [g_app.mainVC mnuEditTestCases];
    }
    else if ([menuItem hasPrefix:@"Add Test Case"]) {
        [g_app.mainVC mnuAddTestCase];
    }
    else if ([menuItem hasPrefix:@"Run Test Cases"]) {
        [self mnuRunTestCases];
    }
    else if ([menuItem hasPrefix:@"Video Mode"]) {
        [self unselectAll];
        [self setState:ITEM_SELECTED forRow:indexPath.row];
        _mode = VIDEO_MODE;
    }
    else if ([menuItem hasPrefix:@"Photo Mode"]) {
        [self unselectAll];
        [self setState:ITEM_SELECTED forRow:indexPath.row];
        _mode = PHOTO_MODE;
    }
    else if ([menuItem hasPrefix:@"Debug Mode"]) {
        [self unselectAll];
        [self setState:ITEM_SELECTED forRow:indexPath.row];
        _mode = DEBUG_MODE;
        [g_app.mainVC debugFlow:true];
    }
    [self.tableView reloadData];
    [topViewController hideLeftViewAnimated:YES completionHandler:nil];
} // didSelectRowAtIndexPath()

// Run all test cases
//-----------------------------------
- (void)mnuRunTestCases
{
    NSArray *testfiles = glob_files(@"", @TESTCASE_PREFIX, @"*.png");
    NSMutableArray *errCounts = [NSMutableArray new];
    for (id fname in testfiles ) {
        NSString *fullfname = getFullPath( fname);
        // Load the image
        UIImage *img = [UIImage imageWithContentsOfFile:fullfname];
        // Load the sgf
        fullfname = changeExtension( fullfname, @".sgf");
        NSString *sgf = [NSString stringWithContentsOfFile:fullfname encoding:NSUTF8StringEncoding error:NULL];
        // Classify
        int nerrs = [g_app.mainVC.cppInterface runTestImg:img withSgf: sgf];
        [errCounts addObject:@(nerrs)];
    } // for
    // Show error counts in a separate View Controller
    UITextView *tv = g_app.testResultsVC.tv;
    NSMutableString *msg = [NSMutableString new];
    [msg appendString:@"Error Count by File\n"];
    [msg appendString:@"===================\n\n"];
    int i = -1;
    for (id fname in testfiles) {
        i++;
        long count = [errCounts[i] integerValue];
        NSString *line = nsprintf( @"%@:\t%5ld\n", fname, count);
        [msg appendString:line];
    }
    tv.text = msg;
    [g_app.navVC pushViewController:g_app.testResultsVC animated:YES];

} // mnuRunTestCases()

@end

































































