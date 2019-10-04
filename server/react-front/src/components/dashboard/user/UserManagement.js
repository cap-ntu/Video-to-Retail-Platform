import React from "react";
import * as PropTypes from "prop-types";
import withStyles from "@material-ui/core/styles/withStyles";
import Sidebar from "./UserSidebar";
import UserDisplay from "./UserDisplay";
import UserEdit from "./UserEdit";
import UserNew from "./UserNew";
import Grow from "@material-ui/core/Grow";
import Fade from "@material-ui/core/Fade";
import MountTransition from "../../common/MountTransition";

const styles = theme => ({
    root: {
        display: "flex",
    },
    textField: {
        marginLeft: theme.spacing.unit,
        marginRight: theme.spacing.unit,
        width: 200,
    },
    section: {
        padding: [[2 * theme.spacing.unit, 0]],
        marginTop: theme.spacing.unit,
        "&:first-child": {
            marginTop: 0,
        }
    },
    sectionInner: {
        paddingTop: 0.5 * theme.spacing.unit,
        "&: first-child": {
            paddingTop: 0,
        }
    },
    cardHeader: {
        display: "flex",
        width: "100%",
        padding: 2 * theme.spacing.unit,
    },
    detail: {
        flex: 1,
        maxWidth: 700,
        margin: "auto",
    },
});

class UserManagement extends React.PureComponent {
    state = {
        edit: false,
        newUser: false,
        currentUser: "",
    };

    componentWillMount() {
        this.props.fetchUserList();
    }

    /**
     * Open creation form
     */
    handleOnCreateUser = () => {
        this.setState({newUser: true, currentUser: ""});
    };

    /**
     *  Send creation form, close form and update sidebar when success
     * @param user new user form
     */
    handleCreateUser = user => {
        const handleCreated = () => {
            this.setState({newUser: false});
            this.props.fetchUserList();
        };
        this.props.handleNewUser(user, handleCreated);
    };

    handleUpdateUser = user => {
        this.props.handleUpdateUser(user, () => this.setState({edit: false}));
    };

    handleDeleteUser = id => {
        this.setState({currentUser: ""});
        this.props.handleDeleteUser(id, this.props.fetchUserList);
    };

    /**
     * Form close factory
     * @param name form flag name
     * @return {Function} form close handler
     */
    handleFormCloseFactory = name => () => {
        this.setState({[`${name}`]: false});
    };

    render() {
        const {classes, users, nameList} = this.props;
        const {edit, newUser, currentUser} = this.state;

        const user = users[users.findIndex(user => user.username === currentUser)];

        return (
            <div className={classes.root}>
                <Sidebar nameList={nameList}
                         currentUser={currentUser}
                         newUser={newUser}
                         handleNewUser={this.handleOnCreateUser}
                         handleClick={user => this.setState({currentUser: user, newUser: false, edit: false})}/>
                <div className={classes.detail}>
                    <MountTransition in={newUser} transition={Grow}>
                        <UserNew
                            classes={{
                                header: classes.cardHeader,
                                textField: classes.textField,
                                section: classes.section,
                                sectionInner: classes.sectionInner,
                            }}
                            handleDone={this.handleCreateUser}
                            handleCancel={() => this.setState({newUser: false})}
                        />
                    </MountTransition>
                    {
                        user ?
                            <React.Fragment>
                                <MountTransition in={edit} transition={Fade} timeout={{exit: 0}}>
                                    <UserEdit classes={{
                                        header: classes.cardHeader,
                                        textField: classes.textField,
                                        section: classes.section,
                                        sectionInner: classes.sectionInner,
                                    }}
                                              user={user}
                                              handleDone={this.handleUpdateUser}
                                              handleCancel={this.handleFormCloseFactory("edit")}
                                              handleDelete={this.handleDeleteUser}/>
                                </MountTransition>
                                <MountTransition in={!edit} transition={Fade} timeout={{exit: 0}}>
                                    <UserDisplay classes={{
                                        header: classes.cardHeader,
                                        section: classes.section,
                                        sectionInner: classes.sectionInner,
                                    }}
                                                 user={user}
                                                 handleEdit={() => this.setState({edit: true})}/>
                                </MountTransition>
                            </React.Fragment> :
                            null
                    }
                </div>
            </div>
        )
    }
}

UserManagement.propTypes = {
    classes: PropTypes.object.isRequired,
    users: PropTypes.arrayOf(
        PropTypes.object.isRequired,
    ).isRequired,
    nameList: PropTypes.object.isRequired,
    fetchUserList: PropTypes.func.isRequired,
    handleNewUser: PropTypes.func.isRequired,
    handleUpdateUser: PropTypes.func.isRequired,
    handleDeleteUser: PropTypes.func.isRequired,
};

export default withStyles(styles)(UserManagement);
