import React from "react";
import * as PropTypes from "prop-types";
import withStyles from "@material-ui/core/styles/withStyles";
import Grid from "@material-ui/core/Grid";
import TextField from "@material-ui/core/TextField";
import Typography from "@material-ui/core/es/Typography/Typography";
import CardContent from "@material-ui/core/es/CardContent/CardContent";
import Button from "../../common/Button";
import Groups from "./formComponent/Groups";
import PermissionControl from "./formComponent/PermissionControl";
import Username from "./formComponent/Username";
import Password from "./formComponent/Password";
import Name from "./formComponent/Name";

const styles = {
    header: {},
    textField: {},
    section: {},
    sectionInner: {},
};

class UserNew extends React.PureComponent {

    state = {
        firstName: "",
        lastName: "",
        username: "",
        password: "",
        _password: "",
        email: "",
        accountType: "",
        domain: "",
        status: "",
        groups: [],
        showPassword: false,
        validation: {
            username: {error: false, rest: undefined},
            password: {error: false, rest: undefined},
            name: {error: false, rest: undefined},
            groups: {error: false, rest: undefined},
            permissionControl: {error: false, rest: undefined},
        }
    };

    child = {
        username: React.createRef(),
        password: React.createRef(),
        name: React.createRef(),
        permissionControl: React.createRef(),
    };

    handleChange = name => event => {
        this.setState({[name]: event.target.value});
    };

    handleSubmit = () => {

        const result = [];

        // call for validate, collect validation
        Object.values(this.child).forEach(v =>
            result.push(v.current.validate()));

        // check errors
        if (result.every(x => !x))
            this.props.handleDone({...this.state, showPassword: undefined, validation: undefined})
    };

    collectValidation = name => _state => {
        this.setState(state => ({...state, validation: {...state.validation, [name]: _state}}));
    };

    render() {
        const {classes, handleCancel} = this.props;
        const {firstName, lastName, username, password, email, groups, accountType, domain, status, validation} = this.state;
        const _validation = {};
        Object.keys(validation).forEach(key => _validation[key] = validation[key].rest);

        return (
            <React.Fragment>
                <div className={classes.header}>
                    <Button color="primary" onClick={handleCancel}>
                        Cancel
                    </Button>
                    <div style={{flexGrow: 1}}/>
                    <Button color="primary" onClick={this.handleSubmit}
                            disabled={!Object.values(validation).every(x => !x.error)}>
                        Done
                    </Button>
                </div>

                <CardContent>
                    <div className={classes.section}>
                        <div className={classes.sectionInner}>
                            <Username classes={{textField: classes.textField}}
                                      innerRef={this.child.username}
                                      value={username}
                                      onChange={this.handleChange("username")}
                                      error={_validation.username}
                                      handleUpdateValidation={this.collectValidation("username")}
                            />
                        </div>
                        <div className={classes.sectionInner}>
                            <Password
                                classes={{textField: classes.textField}}
                                innerRef={this.child.password}
                                value={password}
                                onChange={this.handleChange("password")}
                                error={_validation.password}
                                handleUpdateValidation={this.collectValidation("password")}
                            />
                        </div>
                    </div>
                    <div className={classes.section}>
                        <Typography variant="subtitle1" color="textSecondary">Personal Information</Typography>
                        <Grid direction="row" spacing={8} alignItems="center" container>
                            <Grid item>
                                <Name classes={{textField: classes.textField}}
                                      innerRef={this.child.name}
                                      firstName={firstName}
                                      lastName={lastName}
                                      handleChange={this.handleChange}
                                      error={_validation.name}
                                      handleUpdateValidation={this.collectValidation("name")}
                                />
                            </Grid>
                        </Grid>
                        <div className={classes.sectionInner}>
                            <TextField id="email"
                                       className={classes.textField}
                                       label="Email"
                                       value={email}
                                       onChange={this.handleChange("email")}
                                       margin="dense"/>
                        </div>
                    </div>

                    <div className={classes.section}>
                        <Groups classes={{textField: classes.textField}}
                                groups={groups}
                                onChange={(groups, callback) => this.setState({groups: groups}, callback)}
                                handleUpdateValidation={this.collectValidation("groups")}
                        />
                    </div>

                    <div className={classes.section}>
                        <PermissionControl
                            classes={{textField: classes.textField}}
                            innerRef={this.child.permissionControl}
                            accountType={accountType}
                            domain={domain}
                            status={status}
                            error={_validation.permissionControl}
                            handleChange={this.handleChange}
                            handleUpdateValidation={this.collectValidation("permissionControl")}
                        />
                    </div>
                </CardContent>
            </React.Fragment>
        );
    }
}

UserNew.propTypes = {
    classes: PropTypes.object.isRequired,
    handleCancel: PropTypes.func.isRequired,
    handleDone: PropTypes.func.isRequired,
};

export default withStyles(styles)(UserNew);
