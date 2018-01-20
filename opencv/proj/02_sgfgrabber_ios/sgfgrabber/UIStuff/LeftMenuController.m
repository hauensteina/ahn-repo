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

@interface LeftMenuController ()
@property (strong, nonatomic) NSArray *titlesArray;
@end

@implementation LeftMenuController
// Initialize left menu
//-----------------------
- (id)init
{
    self = [super initWithStyle:UITableViewStylePlain];
    if (self) {
        self.titlesArray = @[@"Add Test Case"
                             ,@"Edit Test Cases"
                             ,@"Run Test Cases"
                             ];

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
    cell.textLabel.text = self.titlesArray[indexPath.row];
    return cell;
} // cellForRowAtIndexPath()

#pragma mark - UITableViewDelegate
//-----------------------------------------------------------------------------------------------
- (CGFloat)tableView:(UITableView *)tableView heightForRowAtIndexPath:(NSIndexPath *)indexPath
{
    return 50;
}
// Handle left menu choice
//--------------------------------------------------------------------------------------------
- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath
{
    TopViewController *topViewController = (TopViewController *)self.sideMenuController;
    NSString *menuItem = _titlesArray[indexPath.row];
    if ([menuItem hasPrefix:@"Edit Test Cases"]) {
        [g_app.mainVC mnuEditTestCases];
    }
    else if ([menuItem hasPrefix:@"Add Test Case"]) {
        [g_app.mainVC mnuAddTestCase];
    }
    else if ([menuItem hasPrefix:@"Run Test Cases"]) {
        [self mnuRunTestCases];
    }
    [topViewController hideLeftViewAnimated:YES completionHandler:nil];
} // didSelectRowAtIndexPath()

// Run all test cases
//-----------------------------------
- (void)mnuRunTestCases
{
    NSArray *testfiles = glob_files(@"", @TESTCASE_PREFIX, @"*.jpg");
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

































































